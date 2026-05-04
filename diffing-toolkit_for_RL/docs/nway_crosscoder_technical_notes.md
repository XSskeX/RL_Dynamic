# N-Way Crosscoder Technical Notes

This document explains the n-way crosscoder changes added on top of the original
pairwise crosscoder path. The goal is to train one shared dictionary over more
than two model checkpoints, so a single feature index can be compared across all
models included in that training run.

## Motivation

The original crosscoder compares two models at the same layer:

```text
base model activation:      [hidden_dim]
finetuned model activation: [hidden_dim]
paired sample:              [2, hidden_dim]
```

This gives one shared feature index inside that pairwise dictionary:

```text
decoder.weight[0, feature_id, :]  # base side
decoder.weight[1, feature_id, :]  # finetuned side
```

However, if a separate pairwise dictionary is trained for each RL checkpoint
pair, feature indices are not guaranteed to align between dictionaries. The
n-way path instead trains one dictionary over multiple checkpoints:

```text
sample: [n_models, hidden_dim]
decoder.weight: [n_models, dict_size, hidden_dim]
```

Within that one dictionary, `feature_id` is shared across all model sides.

## Vendored Dictionary Learning Package

### File

`pyproject.toml`

### Original Logic

The project used the upstream `dictionary-learning` package from Git:

```toml
dictionary-learning = { git = "https://github.com/science-of-finetuning/crosscoder_learning.git" }
```

### New Logic

The project now points to the local vendored package:

```toml
dictionary-learning = { path = "vendor/dictionary_learning", editable = true }
```

This allows local edits to `dictionary_learning` to be used by the project.

### File

`vendor/dictionary_learning/pyproject.toml`

### New Logic

A minimal package descriptor was added so the vendored package can be installed
as an editable local dependency:

```toml
[project]
name = "dictionary-learning"

[tool.setuptools]
packages = ["dictionary_learning", "dictionary_learning.trainers"]

[tool.setuptools.package-dir]
dictionary_learning = "."
```

## Dictionary Learning Compatibility

The vendored library already has generalized crosscoder support through the
`num_layers` argument. In this codebase, the term `num_layers` in
`dictionary_learning` maps to the number of model sides in the crosscoder, not
the transformer layer index.

Relevant shapes:

```text
input x:              [batch, num_layers, activation_dim]
encoder.weight:      [num_layers, activation_dim, dict_size]
decoder.weight:      [num_layers, dict_size, activation_dim]
```

For n-way checkpoint comparison:

```text
num_layers == n_models
```

Because the library already accepts arbitrary `num_layers`, most changes were
made in the project-side configuration, activation loading, training setup, and
analysis.

## New Hydra Method Config

### File

`configs/diffing/method/nway_crosscoder.yaml`

### Original Logic

Only `configs/diffing/method/crosscoder.yaml` existed. It assumed pairwise
crosscoder training and relied on the global `model` plus
`organism.finetuned_models` to resolve:

```text
base model
finetuned model
```

### New Logic

`nway_crosscoder.yaml` mirrors the existing crosscoder training options but adds
an explicit model list:

```yaml
nway:
  run_name: null
  models: []
```

Each entry can be one of:

```yaml
- global_step_30_to_global_step_60
- {config: global_step_30_to_global_step_60, name: step30}
- {name: step30, model_id: /path/to/model}
```

The rest of the config keeps the same training knobs:

```yaml
training:
  expansion_factor: 32
  k: 100
  lr: 1e-4
```

## Model Configuration Resolution

### File

`src/diffing/utils/configs.py`

### Original Logic

`get_model_configurations(cfg)` returned exactly two model configs:

```python
base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)
```

The base config came from `cfg.model`. The finetuned config came from
`cfg.organism.finetuned_models[cfg.model.name][organism_variant]`.

### New Logic

The following helpers were added:

```python
load_model_config_by_name(model_name)
_model_spec_to_config(spec)
get_nway_model_configurations(cfg)
```

`get_nway_model_configurations(cfg)` reads:

```python
cfg.diffing.method.nway.models
```

and returns:

```python
list[ModelConfig]
```

This lets n-way training bypass the pairwise organism lookup and use an explicit
ordered sequence of checkpoints.

## Activation Loading

### File

`src/diffing/utils/activations.py`

### Original Logic

Pairwise activation loading used `PairedActivationCache`:

```python
PairedActivationCache(base_model_cache, finetuned_model_cache, submodule_name)
```

Each sample had shape:

```text
[2, activation_dim]
```

The config path was:

```python
load_activation_datasets_from_config(
    cfg,
    ds_cfgs,
    base_model_cfg,
    finetuned_model_cfg,
    layers,
    split,
)
```

### New Logic

The vendored library already provides `ActivationCacheTuple`, which stacks an
arbitrary number of activation caches.

Added functions:

```python
load_activation_dataset_for_models(...)
load_activation_dataset_for_models_from_config(...)
load_activation_datasets_for_models_from_config(...)
```

These build cache directories for every model in `model_cfgs` and return an
`ActivationCacheTuple`.

Each sample now has shape:

```text
[n_models, activation_dim]
```

## Preprocessing

### File

`src/diffing/pipeline/preprocessing.py`

### Original Logic

Preprocessing always resolved two models:

```python
base_model_cfg, finetuned_model_cfg = get_model_configurations(self.cfg)
```

Then it collected activations for base and finetuned:

```python
self._collect_activations_for_model_dataset(base_model_cfg, dataset_cfg, dataset)
self._collect_activations_for_model_dataset(finetuned_model_cfg, dataset_cfg, dataset)
```

### New Logic

When the method name is `nway_crosscoder`, preprocessing resolves all n-way
models:

```python
model_cfgs = get_nway_model_configurations(self.cfg)
```

Then it loops over every model:

```python
for model_cfg in model_cfgs:
    self._collect_activations_for_model_dataset(model_cfg, dataset_cfg, dataset)
```

For non-n-way methods, the old pairwise logic is preserved.

## Diffing Method Base Class

### File

`src/diffing/methods/diffing_method.py`

### Original Logic

All diffing methods assumed two model configs:

```python
self.base_model_cfg, self.finetuned_model_cfg = get_model_configurations(cfg)
```

### New Logic

For `nway_crosscoder`, the base class now stores:

```python
self.model_cfgs = get_nway_model_configurations(cfg)
self.base_model_cfg = self.model_cfgs[0]
self.finetuned_model_cfg = self.model_cfgs[-1]
```

The first and last models are used as compatibility shims for existing code that
expects `base_model_cfg` and `finetuned_model_cfg`. Pairwise methods still use
the original two-model resolver.

## Pipeline Routing

### File

`src/diffing/pipeline/diffing_pipeline.py`

### Original Logic

Only `crosscoder` routed to `CrosscoderDiffingMethod`:

```python
elif method_name == "crosscoder":
    return CrosscoderDiffingMethod
```

### New Logic

`nway_crosscoder` now routes to the same class:

```python
elif method_name == "crosscoder" or method_name == "nway_crosscoder":
    return CrosscoderDiffingMethod
```

The class internally branches based on `self.method_cfg.name`.

## Crosscoder Method Orchestration

### File

`src/diffing/methods/crosscoder/method.py`

### Original Logic

For each layer, the method built a pairwise run name:

```python
dictionary_name = crosscoder_run_name(
    self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
)
```

Then it trained:

```python
training_metrics, model_path = train_crosscoder_for_layer(...)
```

Analysis computed two self-dot columns:

```text
dec_base_self_dot_ratio_norm
dec_ft_self_dot_ratio_norm
```

### New Logic

The method now detects:

```python
self.is_nway = self.method_cfg.name == "nway_crosscoder"
```

For n-way runs it builds:

```python
dictionary_name = nway_crosscoder_run_name(...)
```

and trains:

```python
training_metrics, model_path = train_nway_crosscoder_for_layer(...)
```

For self-dot analysis, it calls:

```python
update_nway_crosscoder_latent_df_with_self_dot_ratio(...)
```

which emits one self-dot column per model side:

```text
dec_step30_self_dot_ratio_norm
dec_step60_self_dot_ratio_norm
dec_step90_self_dot_ratio_norm
...
```

The original pairwise branch remains unchanged for `crosscoder`.

## Training Dataset Setup

### File

`src/diffing/utils/dictionary/training.py`

### Original Logic

`setup_training_datasets(...)` loaded pairwise activation caches:

```python
caches = load_activation_datasets_from_config(
    cfg,
    ds_cfgs,
    base_model_cfg,
    finetuned_model_cfg,
    layers=[layer],
    split="train",
)
```

Each dataset item had shape:

```text
[2, activation_dim]
```

### New Logic

`setup_nway_training_datasets(...)` was added. It loads:

```python
caches = load_activation_datasets_for_models_from_config(
    cfg,
    ds_cfgs,
    model_cfgs,
    layers=[layer],
    split="train",
)
```

Each dataset item has shape:

```text
[n_models, activation_dim]
```

The rest of the dataset handling is kept parallel to the pairwise path:

- optional `skip_first_n_tokens`
- local shuffling
- train/validation split loading
- sample count allocation
- DataLoader creation

## Normalization

### File

`src/diffing/utils/dictionary/training.py`

### Original Logic

`combine_normalizer(...)` assumed two sides:

```python
running_stats_1 = cache.activation_cache_1.running_stats
running_stats_2 = cache.activation_cache_2.running_stats
mean = torch.stack([running_stats_1.mean, running_stats_2.mean], dim=0)
std = torch.stack([running_stats_1.std(...), running_stats_2.std(...)], dim=0)
```

### New Logic

`combine_normalizer(...)` now supports either `PairedActivationCache` or
`ActivationCacheTuple`.

It extracts component caches generically:

```python
_activation_cache_components(cache)
```

and stacks one mean/std per model side:

```python
mean.shape == [n_models, activation_dim]
std.shape == [n_models, activation_dim]
```

This matches the `NormalizableMixin` shape checks in the vendored crosscoder.

## Trainer Config

### File

`src/diffing/utils/dictionary/training.py`

### Original Logic

`create_crosscoder_trainer_config(...)` implicitly configured a pairwise
crosscoder. The underlying trainer defaulted to `num_layers=2`.

### New Logic

`create_crosscoder_trainer_config(...)` accepts:

```python
model_cfgs: list | None = None
```

When passed, it sets:

```python
"num_layers": len(model_cfgs)
```

and constructs `lm_name` from every model id:

```python
"lm_name": "-".join(model_cfg.model_id for model_cfg in model_cfgs)
```

This is the key line that tells `BatchTopKCrossCoder` to allocate decoder and
encoder tensors for n model sides.

## N-Way Training Function

### File

`src/diffing/utils/dictionary/training.py`

### New Logic

`train_nway_crosscoder_for_layer(...)` was added. It:

1. Resolves `model_cfgs`.
2. Builds n-way train/validation datasets.
3. Checks that a sample has `sample_activation.shape[0] == len(model_cfgs)`.
4. Computes `activation_dim`.
5. Builds a trainer config with `num_layers=len(model_cfgs)`.
6. Calls the existing `trainSAE(...)`.
7. Saves model metrics with:

```json
"training_mode": "nway_crosscoder",
"model_names": [...],
"model_ids": [...]
```

The saved dictionary is still a regular `BatchTopKCrossCoder`, but with
`num_layers == n_models`.

## Self-Dot Analysis

### File

`src/diffing/utils/dictionary/analysis.py`

### Existing Pairwise Logic

The pairwise function reads two decoder slices:

```python
base_decoder_weight = crosscoder.decoder.weight[base_layer]
ft_decoder_weight = crosscoder.decoder.weight[ft_layer]
```

and writes:

```text
dec_base_self_dot_ratio_norm
dec_ft_self_dot_ratio_norm
```

### New N-Way Logic

`update_nway_crosscoder_latent_df_with_self_dot_ratio(...)` loops over every
decoder side:

```python
for model_idx, name in enumerate(model_names):
    ratios = _self_vs_all_dot_ratio(decoder_weight[model_idx])
    latent_df[f"dec_{name}_self_dot_ratio_norm"] = ratios
```

The dataframe keeps one row per feature and one self-dot column per model.

## Current Self-Dot Formula

The helper `_self_vs_all_dot_ratio(...)` now:

1. Unit-normalizes every feature vector.
2. Computes the Gram matrix.
3. Uses absolute dot products in the denominator.

Formula:

```text
u_i = f_i / ||f_i||
self_dot_ratio_i = ||u_i||^2 / sum_j |<u_i, u_j>|
```

Because `u_i` is unit length, the numerator is approximately 1, and the value is
mainly controlled by how much that feature overlaps with all other features.

## Example Usage

With explicit model ids:

```bash
python main.py \
  diffing/method=nway_crosscoder \
  'diffing.method.nway.run_name=rl_steps_30_60_90' \
  'diffing.method.nway.models=[{name:step30,model_id:/share/.../global_step_30/actor_hf_export},{name:step60,model_id:/share/.../global_step_60/actor_hf_export},{name:step90,model_id:/share/.../global_step_90/actor_hf_export}]'
```

With model config files:

```bash
python main.py \
  diffing/method=nway_crosscoder \
  'diffing.method.nway.run_name=rl_steps_30_60_90' \
  'diffing.method.nway.models=[{config:global_step_30_to_global_step_60,name:step30},{config:global_step_60_to_global_step_90,name:step60},{config:global_step_90_to_global_step_120,name:step90}]'
```

## Important Caveats

The n-way path currently focuses on:

- preprocessing activations for all listed models
- loading n-way activation tuples
- training n-way `BatchTopKCrossCoder`
- writing n-way self-dot feature data

Some downstream analysis code still assumes exactly two model sides:

- latent scaling targets such as `base_error` and `ft_error`
- latent steering code paths
- dashboard views that expect pairwise columns
- some code using `base_model_cfg` and `finetuned_model_cfg`

Those paths should be treated as pairwise-only until explicitly generalized.

## Recommended Validation Steps

Before running a large n-way job:

1. Use 3 models only.
2. Use one layer.
3. Lower `num_samples`, `num_validation_samples`, and `max_steps`.
4. Disable optional downstream analysis:

```yaml
analysis:
  latent_scaling:
    enabled: false
  latent_activations:
    enabled: false
  latent_steering:
    enabled: false
```

5. Confirm the saved dictionary has:

```python
model.decoder.weight.shape == (n_models, dict_size, activation_dim)
```

6. Confirm the n-way self-dot CSV has one row per feature and one column per
model side.
