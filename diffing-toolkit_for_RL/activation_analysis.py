from diffing.utils.dictionary import load_dictionary_model
import os
from pathlib import Path
import torch as th
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from torch.nn.functional import cosine_similarity
from diffing.pipeline.diffing_pipeline import DiffingPipeline
from diffing.pipeline.evaluation_pipeline import EvaluationPipeline
from diffing.utils.configs import CONFIGS_DIR
from diffing.utils.configs import get_model_configurations, get_dataset_configurations, get_nway_model_configurations

def activation_analysis():
    model_path = f"{cfg.infrastructure.storage.checkpoint_dir}/{cfg.model.name}/model_final.pt"
    crosscoder = load_dictionary_model(model_path)
    dataset_cfgs = get_dataset_configurations(
        cfg,
        use_chat_dataset=cfg.diffing.method.datasets.use_chat_dataset,
        use_pretraining_dataset=cfg.diffing.method.datasets.use_pretraining_dataset,
        use_training_dataset=cfg.diffing.method.datasets.use_training_dataset,
    )
    model_configs = get_nway_model_configurations(cfg)
    
    caches = load_n_activation_datasets_from_config(
            cfg=cfg,
            ds_cfgs=dataset_cfgs,
            model_cfgs=model_configs,
            layers=[layer],
            split="train",
        )  # Dict {dataset_name: {layer: ActivationCacheTuple, ...}}
    caches = {
        dataset_name: caches[dataset_name][layer] for dataset_name in caches
    }  # Dict {dataset_name: PairedActivationCache}

def _compute_feature_norm(weight):
    
    return th.linalg.vector_norm(weight.float(), ord=2, dim=1)

def compute_feature_norm(crosscoder):
    num_features = crosscoder.decoder.weight.shape[1]
    latent_df = pd.DataFrame(index=range(num_features))
    for i in range(crosscoder.decoder.weight.shape[0]):
        weight = crosscoder.decoder.weight[i]

        ratios = _compute_feature_norm(weight).cpu()

        latent_df[f"dec_{i}_norm"] = ratios.detach().numpy()

    latent_df.to_csv(Path(f"/share/nlp/baijun/shuhan/crosscoder_output/crosscoder_output_for_4/llama32_3B_Instruct_latent_norm.csv"), encoding='utf-8-sig')
    return latent_df

def _compute_feature_drift_cosine(weight, original_weight):
    weight = weight.float()
    original_weight = original_weight.float()
    eps = 1e-8
    similarity = cosine_similarity(weight, original_weight, dim=1, eps=eps)
    return 1.0 - similarity

def compute_feature_drift_cosine(crosscoder):
    num_features = crosscoder.decoder.weight.shape[1]
    latent_df = pd.DataFrame(index=range(num_features))
    original_weight = crosscoder.decoder.weight[0]
    for i in range(1, crosscoder.decoder.weight.shape[0]):
        weight = crosscoder.decoder.weight[i]

        ratios = _compute_feature_drift_cosine(weight, original_weight).cpu()
        latent_df[f"dec_{i}_drift_from_origin"] = ratios.detach().numpy()
    latent_df.to_csv(Path(f"/share/nlp/baijun/shuhan/crosscoder_output/crosscoder_output_for_4/llama32_3B_Instruct_latent_drift.csv"), encoding='utf-8-sig')
    return latent_df



@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    crosscoder = load_dictionary_model(f"{cfg.infrastructure.storage.checkpoint_dir}/{cfg.model.name}/model_final.pt")
    compute_feature_norm(crosscoder)




if __name__ == "__main__":
    main()


