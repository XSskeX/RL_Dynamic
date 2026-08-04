import datasets
from transformers import AutoTokenizer


TRAIN_FILES = [
    "/share/nlp/baijun/shuhan/SynLogic/train.parquet"
]

VAL_FILES = [
'/share/nlp/baijun/shuhan/SynLogic/test.parquet','/share/nlp/baijun/shuhan/MMLU_Pro/test.parquet','/share/nlp/baijun/shuhan/AIME2024/test.parquet','/share/nlp/baijun/shuhan/AIME2025/test.parquet','/share/nlp/baijun/shuhan/AIME2026/test.parquet','/share/nlp/baijun/shuhan/IF_Bench/test.parquet'
]

MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
MAX_PROMPT_LENGTH = 1024


def load_files(files):
    result = []

    for path in files:
        ds = datasets.load_dataset(
            "parquet",
            data_files=path,
            split="train",
        )

        print(f"{path}: {len(ds)} rows")
        result.append(ds)

    return result


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

val_datasets = load_files(VAL_FILES)

merged_val = datasets.concatenate_datasets(val_datasets)

print("合并后 validation 数量:", len(merged_val))
print("合并后列:", merged_val.column_names)
print("合并后 features:")
print(merged_val.features)