import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None

COVERAGE_MAPPING = {
    'Cover-0': 0,
    'Cover-1': 1,
    'Cover-2': 2,
    'Cover-3': 3,
    'Quarters': 4,
    '2-Man': 5,
    'Cover-6': 6
}
NUM_CLASSES = len(COVERAGE_MAPPING)  # 7


class CoverageTransformer(nn.Module):
    def __init__(self, feature_len=5, model_dim=64, num_heads=2, num_layers=4, dim_feedforward=256, dropout=0.1, output_dim=NUM_CLASSES):
        super(CoverageTransformer, self).__init__()
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
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, output_dim),  # 7 output classes
        )

    def forward(self, x):
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.feature_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = self.player_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSORS_DIR = PROJECT_ROOT / "tensors"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

batch_size = 64
learning_rate = 1e-3

week_eval = 8  # train on weeks 1-7, validate on week 8
train_features = torch.load(TENSORS_DIR / f"features_training_week{week_eval}preds.pt", map_location=device)
train_targets = torch.load(TENSORS_DIR / f"targets_training_week{week_eval}preds.pt", map_location=device)
val_features = torch.load(TENSORS_DIR / f"features_val_week{week_eval}preds.pt", map_location=device)
val_targets = torch.load(TENSORS_DIR / f"targets_val_week{week_eval}preds.pt", map_location=device)

print(f"Train samples: {train_features.size(0)}, Val samples: {val_features.size(0)}")


train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(TensorDataset(val_features, val_targets), batch_size=batch_size, shuffle=False, num_workers=0)

model = CoverageTransformer(
    feature_len=5,
    model_dim=64,
    num_heads=2,
    num_layers=4,
    dim_feedforward=64 * 4,
    dropout=0.1,
    output_dim=NUM_CLASSES
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

early_stopping_patience = 5
best_val_loss = float("inf")
epochs_no_improve = 0
num_epochs = 25

train_losses = []
val_losses = []
val_accuracies = []

print()
print("######################### -- COVERAGE CLASSIFIER -- TRAINING #########################")
print()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    model.eval()
    val_running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for val_features_batch, val_targets_batch in val_loader:
            val_features_batch = val_features_batch.to(device)
            val_targets_batch = val_targets_batch.to(device)

            val_outputs = model(val_features_batch)
            val_loss = loss_fn(val_outputs, val_targets_batch)
            val_running_loss += val_loss.item() * val_features_batch.size(0)

            _, predicted = torch.max(val_outputs, 1)
            correct += (predicted == val_targets_batch).sum().item()

    avg_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    val_accuracy = correct / len(val_loader.dataset)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODELS_DIR / "best_model_coverage.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print()
            print("Early stopping triggered. Best version saved under 'best_model_coverage.pth'")
            print()
            break