import os
import re
import subprocess
from pathlib import Path


ROOT = "/share/nlp/baijun/shuhan/ckpt_0"
ROOT_2 = "/share/nlp/baijun/shuhan/ckpt"
MAIN_PY = Path("/share/nlp/baijun/shuhan/RL_Dynamic/diffing-toolkit_for_RL/main.py")

# 你的基础运行参数
COMMON_ARGS = [
    "diffing/method=crosscoder",
]

# 如果你想固定用某个 model config 名称
MODEL_NAME = "llama32_3B_Instruct"


def run_one(base_path: Path, ft_path: Path, run_name: str, verl_to_hf_path: Path):
    cmd = [
        "python",
        "/share/nlp/baijun/shuhan/RL_Dynamic/verl_ckpt_to_hf.py",
        "--actor-dir",
        str(verl_to_hf_path),
        "--output-dir",
        str(ft_path)
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Run failed: {run_name}")

    cmd = [
        "python",
        str(MAIN_PY),
        *COMMON_ARGS,
        f"model.model_id={base_path}/actor_hf_export",
        f"model.name={run_name}",
        f"organism.finetuned_models.{MODEL_NAME}.default.model_id={ft_path}/actor_hf_export",
        f"infrastructure.storage.base_dir=/share/nlp/baijun/shuhan/crosscoder_outputs/{run_name}",
    ]

    print("\n=== Running ===")
    print("base      :", base_path)
    print("finetuned :", ft_path)
    print("run_name  :", run_name)
    print("command   :", " ".join(map(str, cmd)))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Run failed: {run_name}")


def main():

    for i in range(15):
        if (i < 8):
            base_path = Path(ROOT + f"/global_step_{(i + 1) * 30}/actor_hf_export")
        else:
            base_path = Path(ROOT_2 + f"/global_step_{(i + 1) * 30}/actor_hf_export")
        if (i < 7):
            ft_path = Path(ROOT + f"/global_step_{(i + 2) * 30}/actor_hf_export")
            verl_to_hf_path = Path(ROOT + f"/global_step_{(i + 2) * 30}/actor")
        else:
            ft_path = Path(ROOT_2 + f"/global_step_{(i + 2) * 30}/actor_hf_export")
            verl_to_hf_path = Path(ROOT_2 + f"/global_step_{(i + 2) * 30}/actor")

        run_name = f"global_step_{(i + 1) * 30}_to_global_step_{(i + 2) * 30}"
        run_one(base_path, ft_path, run_name, verl_to_hf_path)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
