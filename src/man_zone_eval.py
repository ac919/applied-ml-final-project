from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from transformer_models import ManZoneTransformer


def load_eval_tensors(tensors_dir: Path, week_eval: int = 8):
    features = torch.load(tensors_dir / f"features_val_week{week_eval}preds.pt")
    targets = torch.load(tensors_dir / f"targets_val_manzone_week{week_eval}preds.pt")
    return features, targets


def main():
    tensors_dir = PROJECT_ROOT / "tensors"
    checkpoint_path = PROJECT_ROOT / "models" / "best_model_all_weeks.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training/man_zone_transformer.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    week_eval = 8
    features_all, targets_all = load_eval_tensors(tensors_dir, week_eval)
    eval_loader = DataLoader(
        TensorDataset(features_all, targets_all), batch_size=256, shuffle=False
    )

    model = ManZoneTransformer(
        feature_len=5,
        model_dim=64,
        num_heads=2,
        num_layers=4,
        dim_feedforward=64 * 4,
        dropout=0.1,
        output_dim=2,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_probs_man = []

    with torch.no_grad():
        for features_batch, targets_batch in eval_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            outputs = model(features_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets_batch.cpu().tolist())
            all_probs_man.extend(probs[:, 1].cpu().tolist())

    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / len(all_targets)

    print(f"Evaluation set: week {week_eval} val tensors (trained on weeks 1-7)")
    print(f"Num samples: {len(all_targets)}")
    print(f"Accuracy: {accuracy:.4f}\n")

    # Label mapping in this project is 0=Zone, 1=Man.
    print(classification_report(all_targets, all_preds, target_names=["Zone", "Man"]))
    print("Confusion Matrix [Zone, Man]:")
    print(confusion_matrix(all_targets, all_preds, labels=[0, 1]))
    print(f"AUC-ROC (Man=positive class): {roc_auc_score(all_targets, all_probs_man):.4f}")


if __name__ == "__main__":
    main()