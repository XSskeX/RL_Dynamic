import os
import argparse
import json
import gzip
import struct
import logging
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from delphi.latents import LatentDataset
from delphi.config import ConstructorConfig, SamplerConfig
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

# ----------------------------- Shared Memory Utilities -----------------------------
class SharedLayerActivations:
    """Per-layer shared memory for ragged feature activations"""
    def __init__(self, layer_name, features):
        """
        features: list of dicts with keys "locations": Tensor(Ni,3), "activations": Tensor(Ni,)
        """
        self.layer_name = layer_name
        self.num_features = len(features)

        # Flatten locations and activations
        loc_list = []
        act_list = []
        loc_index = [0]
        act_index = [0]

        for feat in features:
            Ni = feat['locations'].shape[0]
            loc_list.append(feat['locations'].numpy())
            act_list.append(feat['activations'].numpy())
            loc_index.append(loc_index[-1] + Ni)
            act_index.append(act_index[-1] + Ni)

        loc_flat = np.concatenate(loc_list, axis=0).astype(np.int32)
        act_flat = np.concatenate(act_list, axis=0).astype(np.float32)
        self.loc_index = np.array(loc_index, dtype=np.int32)
        self.act_index = np.array(act_index, dtype=np.int32)

        # Create shared memory blocks
        self.loc_shm = shared_memory.SharedMemory(create=True, size=loc_flat.nbytes)
        self.act_shm = shared_memory.SharedMemory(create=True, size=act_flat.nbytes)
        self.loc_shape = loc_flat.shape
        self.act_shape = act_flat.shape

        # Copy data into shared memory
        np_loc = np.ndarray(loc_flat.shape, dtype=loc_flat.dtype, buffer=self.loc_shm.buf)
        np_loc[:] = loc_flat[:]
        np_act = np.ndarray(act_flat.shape, dtype=act_flat.dtype, buffer=self.act_shm.buf)
        np_act[:] = act_flat[:]

    def attach(self):
        """Attach to shared memory in worker process"""
        self.loc_flat = np.ndarray(self.loc_shape, dtype=np.int32, buffer=self.loc_shm.buf)
        self.act_flat = np.ndarray(self.act_shape, dtype=np.float32, buffer=self.act_shm.buf)

    def get_feature(self, feature_idx):
        """Retrieve locations and activations for one feature (as torch tensors)"""
        start_loc, end_loc = self.loc_index[feature_idx], self.loc_index[feature_idx+1]
        start_act, end_act = self.act_index[feature_idx], self.act_index[feature_idx+1]
        loc = torch.from_numpy(self.loc_flat[start_loc:end_loc]).cuda()
        act = torch.from_numpy(self.act_flat[start_act:end_act]).cuda()
        return loc, act

    def cleanup(self):
        self.loc_shm.close()
        self.loc_shm.unlink()
        self.act_shm.close()
        self.act_shm.unlink()


# ----------------------------- Global Variables -----------------------------
global_layers = {}  # key: layer_name -> SharedLayerActivations
global_feature_logits_result = None
global_tokens = None
global_tokenizer = None
global_layer_names_by_idx = None


def init_pool(layers_shm_map, tokens, feature_logits_result, tokenizer, layer_names_by_idx):
    """initializer for ProcessPoolExecutor"""
    global global_layers, global_tokens, global_feature_logits_result, global_tokenizer
    global global_layer_names_by_idx
    global_tokens = tokens
    global_feature_logits_result = feature_logits_result
    global_tokenizer = tokenizer
    global_layer_names_by_idx = layer_names_by_idx
    for layer_name, shared_layer in layers_shm_map.items():
        shared_layer.attach()
        global_layers[layer_name] = shared_layer


def get_feature_data(layer_name, feature_idx):
    layer_shm = global_layers[layer_name]
    loc, act = layer_shm.get_feature(feature_idx)
    return loc, act


# ----------------------------- Feature Extraction Functions -----------------------------
def init_lm_head(model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model.get_output_embeddings()


def get_feature_logits(model_path, base_transcoder_path, layers, num_latents, tokenizer, batch_size=512, topk=5):
    lm_head = init_lm_head(model_path)
    feature_logits_result = {}
    for i in tqdm(layers, desc="Processing layers"):
        feature_logits_result[i] = {}
        transcoder_path = base_transcoder_path.format(layer_idx=i) + '/sae.safetensors'
        transcoder_weight = load_file(transcoder_path)
        sparse_feature_vectors = transcoder_weight['encoder.weight'][:num_latents]
        for j in range(0, num_latents, batch_size):
            batch_sparse_feature_vectors = sparse_feature_vectors[j:j+batch_size].cuda()
            batch_sparse_feature_logits = lm_head(batch_sparse_feature_vectors)
            _, top_indices = torch.topk(batch_sparse_feature_logits, topk, dim=-1, largest=True, sorted=True)
            _, bottom_indices = torch.topk(batch_sparse_feature_logits, topk, dim=-1, largest=False, sorted=True)
            top_indices = top_indices.cpu()
            bottom_indices = bottom_indices.cpu()
            for k in range(len(top_indices)):
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices[k])
                bottom_tokens = tokenizer.convert_ids_to_tokens(bottom_indices[k])
                feature_logits_result[i][j+k] = {"top_logits": top_tokens, "bottom_logits": bottom_tokens}
    return feature_logits_result


def generate_single_feature_data(layer_idx, feature_idx, top_n=10, n_bins=100, random_seed=42):
    layer_name = global_layer_names_by_idx[layer_idx]
    tokens = global_tokens
    feature_logits_result = global_feature_logits_result
    tokenizer = global_tokenizer

    # get ragged loc/act
    loc, acts = get_feature_data(layer_name, feature_idx)
    if loc.shape[0] == 0:
        return None

    xs = loc[:, 0]
    ys = loc[:, 1]

    seq_len = tokens.shape[1]

    all_token_acts = torch.zeros(
        len(torch.unique(xs)), seq_len, device='cuda', dtype=torch.float32
    )
    unique_x, inverse_indices = torch.unique(xs, return_inverse=True)
    all_token_acts[inverse_indices, ys] = acts

    all_act_tokens = tokens[unique_x.cpu()].reshape(-1) 
    all_act_tokens = tokenizer.convert_ids_to_tokens(all_act_tokens) 
    all_act_tokens = [all_act_tokens[i*seq_len:(i+1)*seq_len] for i in range(len(unique_x))]
    
    max_acts, max_indices = all_token_acts.max(dim=1)

    # 按 sample_idx 整理输出
    act_samples = []
    for x in range(len(unique_x)):
        sample = {
            "tokens_acts_list": all_token_acts[x].cpu().tolist(),
            "train_token_ind": max_indices[x].cpu().item(),
            "is_repeated_datapoint": False,
            "tokens": all_act_tokens[x],
        }
        act_samples.append(sample)

    np.random.seed(random_seed)
    max_acts_np = max_acts.cpu().numpy()
    act_max = float(max_acts_np.max())
    act_min = float(max_acts_np.min())
    quantile_values = np.percentile(max_acts_np, np.arange(0, 101, 1)).tolist()
    histogram, _ = np.histogram(max_acts_np, bins=n_bins, range=(act_min, act_max))
    sorted_idx = np.argsort(max_acts_np)[::-1]
    sorted_samples = [act_samples[i] for i in sorted_idx]

    intervals_idx = np.array_split(range(len(sorted_samples)), 7)
    interval_names = ["Top", "Subsample interval 5", "Subsample interval 4",
                      "Subsample interval 3", "Subsample interval 2", "Subsample interval 1", "Bottom"]
    examples_quantiles = []
    for name, idx_list in zip(interval_names, intervals_idx):
        if name == "Top":
            selected_samples = [sorted_samples[j] for j in idx_list[:top_n]]
        elif name == "Bottom":
            selected_samples = [sorted_samples[j] for j in idx_list[-top_n:]]
        else:
            if len(idx_list) <= top_n:
                selected_samples = [sorted_samples[j] for j in idx_list]
            else:
                sampled_idx = np.random.choice(idx_list, size=top_n, replace=False)
                selected_samples = [sorted_samples[j] for j in sampled_idx]
        examples_quantiles.append({"quantile_name": name, "examples": selected_samples})

    activation_frequency = acts.shape[0] / tokens.numel()
    if feature_logits_result is None:
        top_logits = []
        bottom_logits = []
    else:
        logits = feature_logits_result.get(layer_idx, {}).get(feature_idx, {})
        top_logits = logits.get("top_logits", [])
        bottom_logits = logits.get("bottom_logits", [])

    return {
        "transcoder_id": layer_idx,
        "index": feature_idx,
        "examples_quantiles": examples_quantiles,
        "top_logits": top_logits,
        "bottom_logits": bottom_logits,
        "act_max": act_max,
        "act_min": act_min,
        "quantile_values": quantile_values,
        "histogram": histogram.tolist(),
        "activation_frequency": activation_frequency,
    }


def run_single_feature(args):
    layer_idx, feature_idx = args
    return generate_single_feature_data(layer_idx, feature_idx)


# ----------------------------- Main Feature Generation -----------------------------
def generate_feature_files(
    model_path,
    base_transcoder_path,
    latent_dir,
    save_dir,
    layers,
    num_latents,
    overwrite,
    num_workers=4,
    module_template="nway_crosscoder.layer_{layer_idx}",
    skip_feature_logits=False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    hookpoints = [module_template.format(layer_idx=i, layer=i) for i in layers]
    print(f"hookpoint list: {hookpoints}")
    layer_names_by_idx = {
        layer_idx: hookpoint for layer_idx, hookpoint in zip(layers, hookpoints)
    }
    latent_dict = {hp: torch.arange(0, num_latents) for hp in hookpoints}
    latent_dataset = LatentDataset(
        raw_dir=latent_dir,
        modules=hookpoints,
        sampler_cfg=SamplerConfig(),
        constructor_cfg=ConstructorConfig(),
        latents=latent_dict,
        tokenizer=tokenizer
    )
    tokens = latent_dataset.tokens
    all_activations_raw = latent_dataset._load_all_data(None, None)

    # ----------- Prepare shared memory per layer -------------
    layers_shm_map = {}
    print("Preparing activations shared memory...")
    for layer_name in hookpoints:
        feats_dict = all_activations_raw[layer_name]  # dict: feature_idx -> feature_data
        features_list = []

        # 遍历每个 feature_idx，保证顺序和 num_latents 对齐
        for feature_idx in range(num_latents):
            if feature_idx in feats_dict:
                f = feats_dict[feature_idx]
                features_list.append({
                    "locations": f.locations.cpu(),
                    "activations": f.activations.cpu()
                })
            else:
                # 如果该 feature 没有数据，填充空 tensor
                features_list.append({
                    "locations": torch.zeros((0, 3), dtype=torch.int32),
                    "activations": torch.zeros((0,), dtype=torch.float32)
                })

        layers_shm_map[layer_name] = SharedLayerActivations(layer_name, features_list)

    print("Shared memory prepared.")


    # ----------- Compute feature logits -------------
    if skip_feature_logits:
        feature_logits_result = None
        print("Skipping feature logits.")
    else:
        if base_transcoder_path is None:
            raise ValueError(
                "--base_transcoder_path is required unless --skip_feature_logits is set"
            )
        feature_logits_result = get_feature_logits(model_path, base_transcoder_path, layers, num_latents, tokenizer, batch_size=512, topk=5)
        print("Feature logits computed.")

    # ----------- Process features per layer -------------
    for layer_idx in layers:
        feature_indices = list(range(num_latents))
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_pool,
                                 initargs=(layers_shm_map, tokens, feature_logits_result, tokenizer, layer_names_by_idx)) as executor:
            results = list(tqdm(executor.map(run_single_feature, [(layer_idx, i) for i in feature_indices]),
                                total=num_latents, desc=f"Layer {layer_idx}"))

        # 保存结果
        layer_results = [r for r in results if r is not None]
        save_layer_features(save_dir, layer_idx, layer_results, overwrite)
        print(f"Layer {layer_idx} features generated and saved.")

    # ----------- Cleanup shared memory -------------
    for shm in layers_shm_map.values():
        shm.cleanup()


# ----------------------------- Save Function -----------------------------
def save_layer_features(save_dir, layer_idx, features, overwrite=False):
    os.makedirs(save_dir, exist_ok=True)
    bin_filename = f"layer_{layer_idx}.bin"
    bin_path = f"{save_dir}/{bin_filename}"
    offsets = []
    current_offset = 0
    with open(bin_path, "wb") as f:
        for feat in features:
            data = json.dumps(feat).encode("utf-8")
            compressed = gzip.compress(data)
            length_bytes = struct.pack("<I", len(compressed))
            offsets.append(current_offset)
            f.write(length_bytes)
            f.write(compressed)
            current_offset += 4 + len(compressed)
        offsets.append(current_offset)
    index_path = f"{save_dir}/index.json.gz"
    if os.path.exists(index_path) and not overwrite:
        with gzip.open(index_path, "rt", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {'version': '1.0', 'format': 'variable_chunks'}
    index_data[str(layer_idx)] = {"filename": bin_filename, "offsets": offsets}
    with gzip.open(index_path, "wt", encoding="utf-8") as f:
        json.dump(index_data, f)


# ----------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_transcoder_path", type=str, default=None)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--num_latents", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--module_template", type=str, default="nway_crosscoder.layer_{layer_idx}")
    parser.add_argument("--skip_feature_logits", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_feature_files(args.model_path, args.base_transcoder_path,
                           args.latent_dir, args.save_dir,
                           args.layers, args.num_latents, args.overwrite,
                           args.num_workers, args.module_template,
                           args.skip_feature_logits)
    print("Feature generation completed.")


if __name__ == "__main__":
    main()
