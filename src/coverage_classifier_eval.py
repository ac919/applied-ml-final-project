from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix


class CoverageTransformer(nn.Module):
    def __init__(
        self,
        feature_len=5,
        model_dim=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=7,
    ):
        super().__init__()
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )

        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, output_dim),
        )

    def forward(self, x):
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.feature_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = self.player_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
        x = self.decoder(x)
        return x


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
    project_root = Path(__file__).resolve().parent.parent
    tensors_dir = project_root / "tensors"
    checkpoint_path = project_root / "models" / "best_model_coverage.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training/coverage_classifier_transformer.py first."
        )

    week_eval = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_features = torch.load(tensors_dir / f"features_val_week{week_eval}preds.pt", map_location=device)
    val_targets = torch.load(tensors_dir / f"targets_val_week{week_eval}preds.pt", map_location=device)

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