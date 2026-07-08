import json
import re
from pathlib import Path
import pandas as pd

eval_dir = Path("C:/Users/Administrator/Desktop/RL_Data/crosscoder_data/RL_Dynamics_data/outputs/evaluations")

rows = []

for path in eval_dir.glob("*.txt"):
    if path.name == "detection_summary.csv":
        continue

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print("skip", path, e)
        continue

    if not isinstance(data, list) or len(data) == 0:
        continue

    y_true = [x.get("activating") for x in data]
    y_pred = [x.get("prediction") for x in data]

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if len(valid) == 0:
        continue

    y_true_v = [t for t, p in valid]
    y_pred_v = [p for t, p in valid]

    total = len(y_true_v)
    n_pos = sum(bool(x) for x in y_true_v)
    n_neg = total - n_pos

    correct = sum(t == p for t, p in zip(y_true_v, y_pred_v))
    acc = correct / total

    tp = sum(t is True and p is True for t, p in zip(y_true_v, y_pred_v))
    tn = sum(t is False and p is False for t, p in zip(y_true_v, y_pred_v))
    fp = sum(t is False and p is True for t, p in zip(y_true_v, y_pred_v))
    fn = sum(t is True and p is False for t, p in zip(y_true_v, y_pred_v))

    tpr = tp / n_pos if n_pos else None
    tnr = tn / n_neg if n_neg else None
    bal_acc = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None

    m = re.search(r"latent(\d+)", path.name)
    latent_id = int(m.group(1)) if m else None

    rows.append({
        "file": path.name,
        "latent_id": latent_id,
        "total": total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": bal_acc,
        "valid_rate": len(valid) / len(data),
        "all_pred_true": all(p is True for p in y_pred_v),
        "all_pred_false": all(p is False for p in y_pred_v),
    })

df = pd.DataFrame(rows)

good = df[
    (df["total"] >= 10)
    & (df["n_pos"] >= 3)
    & (df["n_neg"] >= 3)
    & (df["valid_rate"] >= 0.8)
    & (df["balanced_accuracy"] >= 0.60)
    & (~df["all_pred_true"])
    & (~df["all_pred_false"])
].sort_values("balanced_accuracy", ascending=False)

df.to_csv(eval_dir / "evaluation_metrics_all.csv", index=False)
good.to_csv(eval_dir / "good_explanations.csv", index=False)

print(good[["file", "latent_id", "total", "n_pos", "n_neg", "accuracy",
"balanced_accuracy"]].head(50))
