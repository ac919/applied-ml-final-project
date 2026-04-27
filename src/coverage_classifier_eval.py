from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from transformer_models import CoverageTransformer

COVERAGE_MAPPING = {
    'Cover-0': 0,
    'Cover-1': 1,
    'Cover-2': 2,
    'Cover-3': 3,
    'Quarters': 4,
    '2-Man': 5,
    'Cover-6': 6
}
CLASS_NAMES = list(COVERAGE_MAPPING.keys())


def main():
    tensors_dir = PROJECT_ROOT / "tensors"
    checkpoint_path = PROJECT_ROOT / "models" / "best_model_coverage.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training/coverage_classifier_transformer.py first."
        )

    week_eval = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_features = torch.load(tensors_dir / f"features_val_week{week_eval}preds.pt", map_location=device)
    val_targets = torch.load(tensors_dir / f"targets_val_coverage_week{week_eval}preds.pt", map_location=device)

    eval_loader = DataLoader(
        TensorDataset(val_features, val_targets), batch_size=256, shuffle=False
    )

    model = CoverageTransformer(
        feature_len=5,
        model_dim=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=len(COVERAGE_MAPPING)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features_batch, targets_batch in eval_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            outputs = model(features_batch)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets_batch.cpu().tolist())

    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / len(all_targets)

    print(f"Evaluation set: week {week_eval} val tensors (trained on weeks 1-7)")
    print(f"Num samples: {len(all_targets)}")
    print(f"Accuracy: {accuracy:.4f}\n")

    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(f"Rows = Actual, Cols = Predicted | {CLASS_NAMES}")
    print(confusion_matrix(all_targets, all_preds))

    # per-class accuracy breakdown
    print("\nPer-class accuracy:")
    for name, idx in COVERAGE_MAPPING.items():
        mask = [t == idx for t in all_targets]
        if sum(mask) == 0:
            continue
        correct = sum(p == t for p, t in zip(all_preds, all_targets) if t == idx)
        print(f"  {name}: {correct}/{sum(mask)} ({100 * correct / sum(mask):.1f}%)")


if __name__ == "__main__":
    main()