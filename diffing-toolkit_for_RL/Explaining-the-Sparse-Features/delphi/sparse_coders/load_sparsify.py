from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Optional, Protocol, Union

import json
import torch
from torch import Tensor, nn
from sparsify import SparseCoderConfig
from sparsify.fused_encoder import fused_encoder
from sparsify.sparse_coder import EncoderOutput
from torch import Tensor
from transformers import PreTrainedModel


from safetensors import safe_open
from safetensors.torch import load_model, save_model


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = SparseCoder(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if not self.cfg.transcode:
            x = x - self.b_dec

        return fused_encoder(
            x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation
        )




class PotentiallyWrappedSparseCoder(Protocol):
    def encode(self, x: Tensor) -> EncoderOutput: ...

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module: ...

    cfg: SparseCoderConfig
    num_latents: int


def sae_dense_latents(x: Tensor, sae: PotentiallyWrappedSparseCoder) -> Tensor:
    """Run `sae` on `x`, yielding the dense activations."""
    x_in = x.reshape(-1, x.shape[-1])
    encoded = sae.encode(x_in)
    buf = torch.zeros(
        x_in.shape[0], sae.num_latents, dtype=x_in.dtype, device=x_in.device
    )
    buf = buf.scatter_(-1, encoded.top_indices, encoded.top_acts.to(buf.dtype))
    return buf.reshape(*x.shape[:-1], -1)


def resolve_path(
    model: PreTrainedModel | torch.nn.Module, path_segments: list[str]
) -> list[str] | None:
    """Attempt to resolve the path segments to the model in the case where it
    has been wrapped (e.g. by a LanguageModel, causal model, or classifier)."""
    # If the first segment is a valid attribute, return the path segments
    if hasattr(model, path_segments[0]):
        return path_segments

    # Look for the first actual model inside potential wrappers
    for attr_name, attr in model.named_children():
        if isinstance(attr, torch.nn.Module):
            print(f"Checking wrapper model attribute: {attr_name}")
            if hasattr(attr, path_segments[0]):
                print(
                    f"Found matching path in wrapper at {attr_name}.{path_segments[0]}"
                )
                return [attr_name] + path_segments

            # Recursively check deeper
            deeper_path = resolve_path(attr, path_segments)
            if deeper_path is not None:
                print(f"Found deeper matching path starting with {attr_name}")
                return [attr_name] + deeper_path
    return None


def load_sparsify_sparse_coders(
    name: str,
    hookpoints: list[str],
    device: str | torch.device,
    compile: bool = False,
) -> dict[str, PotentiallyWrappedSparseCoder]:
    """
    Load sparsify sparse coders for specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to identify the sparse models.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        dict[str, Any]: A dictionary mapping hookpoints to sparse models.
    """

    # Load the sparse models
    sparse_model_dict = {}
    name_path = Path(name)
    if name_path.exists():
        for hookpoint in hookpoints:
            sparse_model_dict[hookpoint] = SparseCoder.load_from_disk(
                name_path / hookpoint, device=device, decoder=False
            )
            if compile:
                sparse_model_dict[hookpoint] = torch.compile(
                    sparse_model_dict[hookpoint]
                )
    else:
        # Load on CPU first to not run out of memory
        sparse_models = SparseCoder.load_many(name, device="cpu")
        for hookpoint in hookpoints:
            sparse_model_dict[hookpoint] = sparse_models[hookpoint].to(device)
            if compile:
                sparse_model_dict[hookpoint] = torch.compile(
                    sparse_model_dict[hookpoint]
                )

        del sparse_models
    return sparse_model_dict


def load_sparsify_hooks(
    model: PreTrainedModel,
    name: str,
    hookpoints: list[str],
    device: str | torch.device | None = None,
    compile: bool = False,
) -> tuple[dict[str, Callable], bool]:
    """
    Load the encode functions for sparsify sparse coders on specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to identify the sparse models.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        dict[str, Callable]: A dictionary mapping hookpoints to encode functions.
    """
    device = model.device or "cpu"
    sparse_model_dict = load_sparsify_sparse_coders(
        name,
        hookpoints,
        device,
        compile,
    )
    hookpoint_to_sparse_encode = {}
    transcode = False
    for hookpoint, sparse_model in sparse_model_dict.items():
        print(f"Resolving path for hookpoint: {hookpoint}")
        path_segments = resolve_path(model, hookpoint.split("."))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")

        hookpoint_to_sparse_encode[".".join(path_segments)] = partial(
            sae_dense_latents, sae=sparse_model
        )
        # We only need to check if one of the sparse models is a transcoder
        if hasattr(sparse_model.cfg, "transcode"):
            if sparse_model.cfg.transcode:
                transcode = True
        if hasattr(sparse_model.cfg, "skip_connection"):
            if sparse_model.cfg.skip_connection:
                transcode = True
    return hookpoint_to_sparse_encode, transcode
