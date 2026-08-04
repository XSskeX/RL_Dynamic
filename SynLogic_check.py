from datasets import load_dataset
from transformers import AutoTokenizer


test_path = "/share/nlp/baijun/shuhan/SynLogic/test.parquet"
model_path = "meta-llama/Llama-3.2-3B-Instruct"

dataset = load_dataset(
    "parquet",
    data_files=test_path,
    split="train",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

# 只读取 prompt，避免其他列干扰检查
prompt_dataset = dataset.select_columns(["prompt"])

lengths = []
failed = []

for idx in range(len(prompt_dataset)):
    try:
        prompt = prompt_dataset[idx]["prompt"]

        token_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        lengths.append((idx, len(token_ids)))

    except Exception as exc:
        failed.append(
            {
                "index": idx,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

num_valid = sum(length <= 1024 for _, length in lengths)
num_overlong = sum(length > 1024 for _, length in lengths)

print("总样本数:", len(prompt_dataset))
print("成功计算长度:", len(lengths))
print("读取或分词失败:", len(failed))
print("<= 1024:", num_valid)
print("> 1024:", num_overlong)

if lengths:
    sorted_lengths = sorted(length for _, length in lengths)

    print("最短:", sorted_lengths[0])
    print("最长:", sorted_lengths[-1])
    print("平均:", sum(sorted_lengths) / len(sorted_lengths))

    print("\n最长的 20 条:")
    for idx, length in sorted(
        lengths,
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(f"index={idx}, tokens={length}")

if failed:
    print("\n前 20 个读取失败样本:")
    for item in failed[:20]:
        print(item)