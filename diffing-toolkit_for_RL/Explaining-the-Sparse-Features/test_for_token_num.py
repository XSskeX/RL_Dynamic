from transformers import AutoTokenizer
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from delphi.explainers import DefaultExplainer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

latent_dataset = LatentDataset(
    raw_dir="/share/nlp/baijun/shuhan/model-organisms-for-4/diffing_results/nway_crosscoder_delphi_cache",
    modules=["nway_crosscoder.layer_13"],
    sampler_cfg=SamplerConfig(
        n_examples_train=20,
        n_examples_test=20,
        n_quantiles=10,
        train_type="quantiles",
        test_type="quantiles",
    ),
    constructor_cfg=ConstructorConfig(
        min_examples=100,
        example_ctx_len=128,
        non_activating_source="random",
    ),
    latents={"nway_crosscoder.layer_13": torch.arange(0, 100)},
    tokenizer=tokenizer,
)

explainer = DefaultExplainer(
    client=None,
    threshold=0.1,
    activations=True,
    cot=False,
    sentence_level=False,
)

def count_messages(messages):
    text = ""
    for m in messages:
        text += f"{m['role']}:\n{m['content']}\n"
    return len(tokenizer(text).input_ids)

counts = []
for i, record in enumerate(latent_dataset):
    if i >= 20:
        break
    messages = explainer._build_prompt(record.train)
    counts.append(count_messages(messages))

print("avg input tokens:", sum(counts) / len(counts))
print("min:", min(counts), "max:", max(counts))