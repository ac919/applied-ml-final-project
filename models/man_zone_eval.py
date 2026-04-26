from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class ManZoneTransformer(nn.Module):
    def __init__(
        self,
        feature_len=5,
        model_dim=64,
        num_heads=2,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=2,
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


def load_eval_tensors(tensors_dir: Path, week_eval: int = 8):
    features = torch.load(tensors_dir / f"features_val_week{week_eval}preds.pt")
    targets = torch.load(tensors_dir / f"targets_val_week{week_eval}preds.pt")
    return features, targets


def main():
    project_root = Path(__file__).resolve().parent.parent
    tensors_dir = project_root / "tensors"
    checkpoint_path = project_root / "models" / "best_model_all_weeks.pth"

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