"""Model-axis tensor parallelism for N-way BatchTopK CrossCoders.

This module intentionally contains only the sharded model primitives.  It does
not modify or hook the existing training loop.  A later integration can select
``ModelShardedBatchTopKCrossCoder`` from the CrossCoder trainer when tensor
parallelism is enabled.

The sharding axis is called ``num_layers`` in the original CrossCoder code, but
for an N-way CrossCoder it represents the model axis.  Each device owns the
encoder and decoder matrices for a contiguous subset of those models:

    encoder weight: [local_models, activation_dim, dict_size]
    decoder weight: [local_models, dict_size, activation_dim]

The latent space is not sharded.  Encoder partial sums are reduced onto the
root device, BatchTopK is evaluated once on that device, and the resulting
latent tensor is copied to each decoder shard.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .dictionary import BatchTopKCrossCoder, CodeNormalization


@dataclass(frozen=True)
class ModelShardSpec:
    """Describe the contiguous model range assigned to one device."""

    start: int
    stop: int
    device: torch.device

    @property
    def size(self) -> int:
        """Return the number of models assigned to this shard."""

        return self.stop - self.start

    @property
    def model_indices(self) -> range:
        """Return the global model indices owned by this shard."""

        return range(self.start, self.stop)


def partition_model_axis(
    num_models: int,
    devices: Sequence[str | torch.device],
) -> tuple[ModelShardSpec, ...]:
    """Split the model axis into balanced contiguous device assignments.

    Earlier devices receive one extra model when ``num_models`` is not evenly
    divisible by the number of devices.  Contiguous ranges make batch slicing
    and canonical checkpoint reconstruction straightforward.
    """

    if num_models <= 0:
        raise ValueError(f"num_models must be positive, got {num_models}")
    if not devices:
        raise ValueError("At least one tensor-parallel device is required")
    if len(devices) > num_models:
        raise ValueError(
            "The number of tensor-parallel devices cannot exceed the number "
            f"of models: {len(devices)} devices for {num_models} models"
        )

    normalized_devices = tuple(torch.device(device) for device in devices)
    base_size, remainder = divmod(num_models, len(normalized_devices))

    specs: list[ModelShardSpec] = []
    start = 0
    for shard_idx, device in enumerate(normalized_devices):
        shard_size = base_size + int(shard_idx < remainder)
        stop = start + shard_size
        specs.append(ModelShardSpec(start=start, stop=stop, device=device))
        start = stop

    return tuple(specs)


class CrossCoderModelShard(nn.Module):
    """Own and compute with the parameters for a subset of N-way models."""

    def __init__(
        self,
        spec: ModelShardSpec,
        *,
        encoder_weight: torch.Tensor,
        decoder_weight: torch.Tensor,
        decoder_bias: torch.Tensor,
        activation_mean: torch.Tensor,
        activation_std: torch.Tensor,
        activation_global_scale: torch.Tensor,
        has_activation_normalizer: bool,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.has_activation_normalizer = has_activation_normalizer

        self.encoder_weight = nn.Parameter(
            encoder_weight.detach().clone().to(spec.device)
        )
        self.decoder_weight = nn.Parameter(
            decoder_weight.detach().clone().to(spec.device)
        )
        self.decoder_bias = nn.Parameter(
            decoder_bias.detach().clone().to(spec.device)
        )

        self.register_buffer(
            "activation_mean", activation_mean.detach().clone().to(spec.device)
        )
        self.register_buffer(
            "activation_std", activation_std.detach().clone().to(spec.device)
        )
        self.register_buffer(
            "activation_global_scale",
            activation_global_scale.detach().clone().to(spec.device),
        )

        if self.encoder_weight.ndim != 3:
            raise ValueError("encoder_weight must have shape [models, D, F]")
        if self.decoder_weight.ndim != 3:
            raise ValueError("decoder_weight must have shape [models, F, D]")
        if self.encoder_weight.shape[0] != spec.size:
            raise ValueError("encoder_weight model dimension does not match shard size")
        if self.decoder_weight.shape[0] != spec.size:
            raise ValueError("decoder_weight model dimension does not match shard size")

    @property
    def device(self) -> torch.device:
        """Return the device on which this shard's parameters live."""

        return self.encoder_weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the parameter dtype used by this shard."""

        return self.encoder_weight.dtype

    def normalize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a local activation slice using local model statistics."""

        if not self.has_activation_normalizer:
            return x
        return (x - self.activation_mean) * self.activation_global_scale.unsqueeze(
            0
        ).unsqueeze(-1)

    def denormalize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Invert local activation normalization for reconstructed activations."""

        if not self.has_activation_normalizer:
            return x
        x = x / (
            self.activation_global_scale.unsqueeze(0).unsqueeze(-1) + 1e-8
        )
        return x + self.activation_mean

    def encode_partial(
        self,
        x: torch.Tensor,
        *,
        normalize_activations: bool = True,
    ) -> torch.Tensor:
        """Compute this shard's contribution to the shared encoder preactivation.

        The shared encoder bias and ReLU are deliberately omitted.  They must be
        applied exactly once after partial results from every shard are summed.
        """

        if x.ndim != 3:
            raise ValueError("x must have shape [batch, local_models, activation_dim]")
        if x.shape[1] != self.spec.size:
            raise ValueError(
                f"Expected {self.spec.size} local models, got activation shape {x.shape}"
            )
        if normalize_activations:
            x = self.normalize_activations(x)
        return torch.einsum("bld,ldf->bf", x, self.encoder_weight)

    def decode_local(
        self,
        f: torch.Tensor,
        *,
        add_bias: bool = True,
        denormalize_activations: bool = False,
    ) -> torch.Tensor:
        """Decode shared latent features into this shard's model activations."""

        if f.ndim == 2:
            x_hat = torch.einsum("bf,lfd->bld", f, self.decoder_weight)
        elif f.ndim == 3:
            if f.shape[1] != self.spec.size:
                raise ValueError(
                    "Per-model latent input does not match the local shard size"
                )
            x_hat = torch.einsum("blf,lfd->bld", f, self.decoder_weight)
        else:
            raise ValueError("f must have shape [batch, F] or [batch, models, F]")

        if add_bias:
            x_hat = x_hat + self.decoder_bias
        if denormalize_activations:
            x_hat = self.denormalize_activations(x_hat)
        return x_hat

    def decoder_norms(self) -> torch.Tensor:
        """Return per-model, per-feature decoder vector norms."""

        return self.decoder_weight.norm(dim=2)


class ModelShardedBatchTopKCrossCoder(nn.Module):
    """Training-time model-axis sharding for a BatchTopK CrossCoder.

    Parameters are initialized by slicing a canonical ``BatchTopKCrossCoder``.
    This guarantees identical initialization and gives a simple correctness
    oracle while the tensor-parallel trainer is being integrated.

    The class currently rejects DECOUPLED code normalization because that mode
    carries a separate latent mask and threshold for every model.
    """

    _SHARDED_STATE_KEYS = {
        "encoder.weight",
        "encoder.bias",
        "decoder.weight",
        "decoder.bias",
        "activation_mean",
        "activation_std",
        "activation_global_scale",
    }

    def __init__(
        self,
        reference_model: BatchTopKCrossCoder,
        devices: Sequence[str | torch.device],
    ) -> None:
        super().__init__()
        if not isinstance(reference_model, BatchTopKCrossCoder):
            raise TypeError(
                "reference_model must be an instance of BatchTopKCrossCoder"
            )
        if reference_model.decoupled_code:
            raise NotImplementedError(
                "DECOUPLED code normalization is not supported by model-axis "
                "tensor parallelism yet"
            )
        if reference_model.decoder.num_layers != reference_model.num_layers:
            raise ValueError(
                "Tensor parallelism currently requires the same number of "
                "encoder and decoder model slices"
            )

        self.num_models = reference_model.num_layers
        # Keep the original name for compatibility with existing trainer code.
        self.num_layers = reference_model.num_layers
        self.activation_dim = reference_model.activation_dim
        self.dict_size = reference_model.dict_size
        self.code_normalization = reference_model.code_normalization
        self.code_normalization_alpha_sae = (
            reference_model.code_normalization_alpha_sae
        )
        self.code_normalization_alpha_cc = (
            reference_model.code_normalization_alpha_cc
        )
        self.decoupled_code = False
        self.shard_specs = partition_model_axis(self.num_models, devices)
        self.root_device = self.shard_specs[0].device

        self.encoder_bias = nn.Parameter(
            reference_model.encoder.bias.detach().clone().to(self.root_device)
        )
        self.register_buffer(
            "k", reference_model.k.detach().clone().to(self.root_device)
        )
        self.register_buffer(
            "threshold",
            reference_model.threshold.detach().clone().to(self.root_device),
        )
        self.register_buffer(
            "target_rms",
            reference_model.target_rms.detach().clone().to(self.root_device),
        )
        self.register_buffer(
            "code_normalization_id",
            reference_model.code_normalization_id.detach()
            .clone()
            .to(self.root_device),
        )

        has_normalizer = reference_model.has_activation_normalizer
        shards: list[CrossCoderModelShard] = []
        for spec in self.shard_specs:
            model_slice = slice(spec.start, spec.stop)
            shards.append(
                CrossCoderModelShard(
                    spec,
                    encoder_weight=reference_model.encoder.weight[model_slice],
                    decoder_weight=reference_model.decoder.weight[model_slice],
                    decoder_bias=reference_model.decoder.bias[model_slice],
                    activation_mean=reference_model.activation_mean[model_slice],
                    activation_std=reference_model.activation_std[model_slice],
                    activation_global_scale=reference_model.activation_global_scale[
                        model_slice
                    ],
                    has_activation_normalizer=has_normalizer,
                )
            )
        self.shards = nn.ModuleList(shards)

        # Preserve small/future canonical state entries without retaining full
        # unsharded weights in host memory.
        self._canonical_extra_state = OrderedDict(
            (
                key,
                value.detach().clone().cpu(),
            )
            for key, value in reference_model.state_dict().items()
            if key not in self._SHARDED_STATE_KEYS
        )

    @property
    def device(self) -> torch.device:
        """Return the root device used for shared latent computations."""

        return self.root_device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the shared encoder bias and model parameters."""

        return self.encoder_bias.dtype

    def set_dtype(self, dtype: torch.dtype) -> "ModelShardedBatchTopKCrossCoder":
        """Cast all shards without changing their device placement."""

        self.to(dtype=dtype)
        return self

    def set_k(self, k: int) -> None:
        """Update the BatchTopK sparsity target in place."""

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k.fill_(k)

    def prepare_batch(
        self,
        x: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Slice a canonical batch and transfer each model slice to its device."""

        if x.ndim != 3:
            raise ValueError("x must have shape [batch, num_models, activation_dim]")
        if x.shape[1] != self.num_models or x.shape[2] != self.activation_dim:
            raise ValueError(
                "Expected batch shape [B, "
                f"{self.num_models}, {self.activation_dim}], got {tuple(x.shape)}"
            )

        target_dtype = self.dtype if dtype is None else dtype
        return tuple(
            x[:, spec.start : spec.stop, :]
            .contiguous()
            .to(spec.device, dtype=target_dtype, non_blocking=non_blocking)
            for spec in self.shard_specs
        )

    def normalize_batch(
        self, batch_shards: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Apply the appropriate local normalizer to every activation shard."""

        self._validate_batch_shards(batch_shards)
        return tuple(
            shard.normalize_activations(x_local)
            for shard, x_local in zip(self.shards, batch_shards)
        )

    def encoder_preactivation(
        self,
        batch_shards: Sequence[torch.Tensor],
        *,
        normalize_activations: bool = True,
    ) -> torch.Tensor:
        """Reduce local encoder contributions and add the shared encoder bias."""

        self._validate_batch_shards(batch_shards)
        preactivation: torch.Tensor | None = None
        for shard, x_local in zip(self.shards, batch_shards):
            partial = shard.encode_partial(
                x_local,
                normalize_activations=normalize_activations,
            ).to(self.root_device)
            preactivation = (
                partial if preactivation is None else preactivation + partial
            )

        if preactivation is None:  # Defensive; construction requires >= 1 shard.
            raise RuntimeError("No tensor-parallel shards are available")
        return preactivation + self.encoder_bias

    def get_code_normalization(self) -> torch.Tensor:
        """Compute the exact global decoder-based feature normalization.

        Device-to-device copies remain in the autograd graph, so decoder weights
        receive gradients through code normalization just as in the unsharded
        implementation.
        """

        local_norms = [
            shard.decoder_norms().to(self.root_device) for shard in self.shards
        ]

        if self.code_normalization == CodeNormalization.SAE:
            squared_sum = sum(norm.square().sum(dim=0) for norm in local_norms)
            return squared_sum.sqrt().unsqueeze(0)

        if self.code_normalization == CodeNormalization.CROSSCODER:
            return sum(norm.sum(dim=0) for norm in local_norms).unsqueeze(0)

        if self.code_normalization == CodeNormalization.MIXED:
            squared_sum = sum(norm.square().sum(dim=0) for norm in local_norms)
            sae_norm = squared_sum.sqrt().unsqueeze(0)
            crosscoder_norm = sum(
                norm.sum(dim=0) for norm in local_norms
            ).unsqueeze(0)
            return (
                sae_norm * self.code_normalization_alpha_sae
                + crosscoder_norm * self.code_normalization_alpha_cc
            )

        if self.code_normalization == CodeNormalization.NONE:
            return torch.ones(
                (1, self.dict_size),
                device=self.root_device,
                dtype=self.dtype,
            )

        raise NotImplementedError(
            f"Unsupported code normalization: {self.code_normalization}"
        )

    def encode(
        self,
        batch_shards: Sequence[torch.Tensor],
        *,
        return_active: bool = False,
        use_threshold: bool = True,
        normalize_activations: bool = True,
    ):
        """Encode all model shards and apply one global BatchTopK operation."""

        preactivation = self.encoder_preactivation(
            batch_shards,
            normalize_activations=normalize_activations,
        )
        post_relu_f = torch.relu(preactivation)
        code_normalization = self.get_code_normalization()
        post_relu_f_scaled = post_relu_f * code_normalization

        if use_threshold:
            f = post_relu_f * (post_relu_f_scaled > self.threshold)
        else:
            batch_size = post_relu_f.shape[0]
            num_selected = int(self.k.item()) * batch_size
            flattened_scaled = post_relu_f_scaled.flatten()
            if num_selected > flattened_scaled.numel():
                raise ValueError(
                    f"BatchTopK requested {num_selected} entries from a tensor "
                    f"with only {flattened_scaled.numel()} entries"
                )
            topk_indices = flattened_scaled.topk(
                num_selected, sorted=False, dim=-1
            ).indices
            selected_values = post_relu_f.flatten()[topk_indices]
            f = (
                torch.zeros_like(flattened_scaled)
                .scatter_(-1, topk_indices, selected_values)
                .reshape_as(post_relu_f)
            )

        if not return_active:
            return f

        f_scaled = f * code_normalization
        active_features = f.sum(dim=0) > 0
        return (
            f,
            f_scaled,
            active_features,
            post_relu_f,
            post_relu_f_scaled,
        )

    def decode(
        self,
        f: torch.Tensor,
        *,
        add_bias: bool = True,
        denormalize_activations: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Broadcast shared latents and decode one reconstruction per shard."""

        return tuple(
            shard.decode_local(
                f.to(shard.device),
                add_bias=add_bias,
                denormalize_activations=denormalize_activations,
            )
            for shard in self.shards
        )

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        *,
        output_features: bool = False,
        use_threshold: bool = True,
        normalize_activations: bool = True,
    ):
        """Run a sharded encode/decode pass without gathering reconstructions."""

        batch_shards = self.prepare_batch(x) if isinstance(x, torch.Tensor) else x
        f = self.encode(
            batch_shards,
            use_threshold=use_threshold,
            normalize_activations=normalize_activations,
        )
        x_hat_shards = self.decode(
            f,
            denormalize_activations=normalize_activations,
        )
        if output_features:
            return x_hat_shards, f * self.get_code_normalization()
        return x_hat_shards

    @torch.no_grad()
    def decoder_norms(
        self, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Gather only decoder norms in canonical model order for logging."""

        output_device = torch.device(device)
        return torch.cat(
            [shard.decoder_norms().to(output_device) for shard in self.shards],
            dim=0,
        )

    @torch.no_grad()
    def canonical_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Reconstruct a standard unsharded CrossCoder state dict on CPU.

        The returned keys and tensor shapes are compatible with
        ``BatchTopKCrossCoder.from_pretrained``.  Tensors are preallocated and
        filled shard by shard to avoid the extra host-memory peak of ``cat``.
        """

        state = OrderedDict(
            (key, value.detach().clone().cpu())
            for key, value in self._canonical_extra_state.items()
        )

        state["encoder.weight"] = torch.empty(
            (self.num_models, self.activation_dim, self.dict_size),
            dtype=self.dtype,
            device="cpu",
        )
        state["decoder.weight"] = torch.empty(
            (self.num_models, self.dict_size, self.activation_dim),
            dtype=self.dtype,
            device="cpu",
        )
        state["decoder.bias"] = torch.empty(
            (self.num_models, self.activation_dim),
            dtype=self.dtype,
            device="cpu",
        )

        normalization_dtype = self.shards[0].activation_mean.dtype
        state["activation_mean"] = torch.empty(
            (self.num_models, self.activation_dim),
            dtype=normalization_dtype,
            device="cpu",
        )
        state["activation_std"] = torch.empty_like(state["activation_mean"])
        state["activation_global_scale"] = torch.empty(
            (self.num_models,),
            dtype=self.shards[0].activation_global_scale.dtype,
            device="cpu",
        )

        for shard in self.shards:
            model_slice = slice(shard.spec.start, shard.spec.stop)
            state["encoder.weight"][model_slice].copy_(
                shard.encoder_weight.detach().cpu()
            )
            state["decoder.weight"][model_slice].copy_(
                shard.decoder_weight.detach().cpu()
            )
            state["decoder.bias"][model_slice].copy_(
                shard.decoder_bias.detach().cpu()
            )
            state["activation_mean"][model_slice].copy_(
                shard.activation_mean.detach().cpu()
            )
            state["activation_std"][model_slice].copy_(
                shard.activation_std.detach().cpu()
            )
            state["activation_global_scale"][model_slice].copy_(
                shard.activation_global_scale.detach().cpu()
            )

        state["encoder.bias"] = self.encoder_bias.detach().cpu().clone()

        # Reassign shared entries so the exported checkpoint always reflects
        # their current training values rather than constructor-time values.
        state["k"] = self.k.detach().cpu().clone()
        state["threshold"] = self.threshold.detach().cpu().clone()
        state["target_rms"] = self.target_rms.detach().cpu().clone()
        state["code_normalization_id"] = (
            self.code_normalization_id.detach().cpu().clone()
        )
        return state

    @torch.no_grad()
    def materialize_cpu_model(self) -> BatchTopKCrossCoder:
        """Build a normal CPU BatchTopKCrossCoder from the sharded parameters."""

        model = BatchTopKCrossCoder(
            activation_dim=self.activation_dim,
            dict_size=self.dict_size,
            num_layers=self.num_models,
            k=int(self.k.item()),
            code_normalization=self.code_normalization,
            code_normalization_alpha_sae=self.code_normalization_alpha_sae,
            code_normalization_alpha_cc=self.code_normalization_alpha_cc,
        )
        model.load_state_dict(self.canonical_state_dict())
        return model.to(dtype=self.dtype, device="cpu")

    def _validate_batch_shards(
        self, batch_shards: Sequence[torch.Tensor]
    ) -> None:
        """Validate count, shape, and placement of prepared activation shards."""

        if len(batch_shards) != len(self.shards):
            raise ValueError(
                f"Expected {len(self.shards)} batch shards, got {len(batch_shards)}"
            )
        batch_size: int | None = None
        for shard, x_local in zip(self.shards, batch_shards):
            expected_tail = (shard.spec.size, self.activation_dim)
            if x_local.ndim != 3 or tuple(x_local.shape[1:]) != expected_tail:
                raise ValueError(
                    "Invalid local activation shape: expected [B, "
                    f"{expected_tail[0]}, {expected_tail[1]}], got {tuple(x_local.shape)}"
                )
            if x_local.device != shard.device:
                raise ValueError(
                    f"Shard {shard.spec.start}:{shard.spec.stop} is on "
                    f"{shard.device}, but its input is on {x_local.device}"
                )
            if batch_size is None:
                batch_size = x_local.shape[0]
            elif x_local.shape[0] != batch_size:
                raise ValueError("All activation shards must use the same batch size")


@torch.no_grad()
def global_grad_norm(
    parameters: Iterable[nn.Parameter],
    root_device: str | torch.device,
) -> torch.Tensor:
    """Compute the global L2 gradient norm across parameters on many devices."""

    root_device = torch.device(root_device)
    total_squared = torch.zeros((), dtype=torch.float32, device=root_device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total_squared.add_(grad.detach().float().square().sum().to(root_device))
    return total_squared.sqrt()


@torch.no_grad()
def clip_grad_norm_across_devices(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    root_device: str | torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradients using one norm computed across every parameter shard.

    Returns the norm before clipping, matching ``torch.nn.utils.clip_grad_norm_``.
    """

    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    parameters = tuple(parameters)
    total_norm = global_grad_norm(parameters, root_device)
    clip_coefficient = (max_norm / (total_norm + eps)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(clip_coefficient.to(parameter.grad.device))
    return total_norm
