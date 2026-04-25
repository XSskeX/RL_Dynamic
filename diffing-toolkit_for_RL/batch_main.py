import os
import re
import subprocess
from pathlib import Path
import argparse

ROOT = "/share/nlp/baijun/shuhan/ckpt_0"
ROOT_2 = "/share/nlp/baijun/shuhan/ckpt"
MAIN_PY = Path("/share/nlp/baijun/shuhan/RL_Dynamic/diffing-toolkit_for_RL/main.py")

COMMON_ARGS = [
    "diffing/method=crosscoder",
]



def run_one(base_path: Path, ft_path: Path, run_name: str, verl_to_hf_path: Path, activation_dir: Path = None):
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
        f"model={run_name}",
    ]

    print("\n=== Running ===")
    print("base      :", base_path)
    print("finetuned :", ft_path)
    print("run_name  :", run_name)
    print("command   :", " ".join(map(str, cmd)))

    result = subprocess.run(cmd, check=False)

    cmd = [
        "rm",
        "-rf",
        str(activation_dir)
    ]
    if result.returncode != 0:
        raise RuntimeError(f"Run failed: {run_name}")


def run_batch_main(start, end):
    start = start / 30
    end = end / 30
    for i in range(start, end):
        if (i < 9):
            base_path = Path(ROOT + f"/global_step_{i * 30}/actor_hf_export")
        else:
            base_path = Path(ROOT_2 + f"/global_step_{i * 30}/actor_hf_export")
        if (i < 8):
            ft_path = Path(ROOT + f"/global_step_{(i + 1) * 30}/actor_hf_export")
            verl_to_hf_path = Path(ROOT + f"/global_step_{(i + 1) * 30}/actor")
        else:
            ft_path = Path(ROOT_2 + f"/global_step_{(i + 1) * 30}/actor_hf_export")
            verl_to_hf_path = Path(ROOT_2 + f"/global_step_{(i + 1) * 30}/actor")

        run_name = f"global_step_{i * 30}_to_global_step_{(i + 1) * 30}"
        run_one(base_path, ft_path, run_name, verl_to_hf_path)
        activation_dir = Path(f"/share/nlp/baijun/shuhan/model-organisms/activations/global_step_{i * 30}")
    print("\nAll runs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=15)
    args = parser.parse_args()
    run_batch_main(args.start, args.end)