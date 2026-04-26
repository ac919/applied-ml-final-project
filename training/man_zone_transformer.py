import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None

class ManZoneTransformer(nn.Module):
  def __init__(self, feature_len=5, model_dim=64, num_heads=2, num_layers=4, dim_feedforward=256, dropout=0.1, output_dim=2):
      super(ManZoneTransformer, self).__init__()
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
          nn.Linear(model_dim // 4, output_dim),
      )

  def forward(self, x):
      # x shape: (batch_size, num_players, feature_len)
      x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
      x = self.feature_embedding_layer(x)
      x = self.transformer_encoder(x)
      x = self.player_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
      x = self.decoder(x)
      return x
  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSORS_DIR = PROJECT_ROOT / "tensors"
MODELS_DIR = PROJECT_ROOT / "models" 
MODELS_DIR.mkdir(parents=True, exist_ok=True)

batch_size = 64
learning_rate = 1e-3

weeks_train = list(range(1, 9))

for week_eval in weeks_train:

  # loading in data & placing into DataLoader object

    train_features = torch.load(TENSORS_DIR / f"features_training_week{week_eval}preds.pt")
    train_targets = torch.load(TENSORS_DIR / f"targets_training_week{week_eval}preds.pt")

    val_features = torch.load(TENSORS_DIR / f"features_val_week{week_eval}preds.pt")
    val_targets = torch.load(TENSORS_DIR / f"targets_val_week{week_eval}preds.pt")

  # move data to device (think it needs to be consistent)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

  # defining ManZoneTransformer params, initializing optimizer and loss_fn
    model = ManZoneTransformer(
        feature_len=5,    # num of input features (x, y, v_x, v_y, defense)
        model_dim=64,     # experimented with 96 & 128... seems best
        num_heads=2,      # 2 seems best (but may have overfit when tried 4... may be worth iterating & increasing dropout)
        num_layers=4,
        dim_feedforward=64 * 4,
        dropout=0.1,      # 10% dropout to prevent overfitting... iterate as model becomes more complex (industry std is higher, i believe)
        output_dim=2      # man or zone classification
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

  # manually placing an early stopping method... will iterate on the exact value (currently 5) but want to prevent overfitting
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

  # -- believe industry standard is closer to 50, suggests leaving room on table to grid search over hyperparams (but lacking the compute for that)
    num_epochs = 30 # keeping a higher mark... ~15-20 was best in previous training but early stopping should prevent overfitting...

    train_losses = []
    val_losses = []
    val_accuracies = []

    print()
    print(f"######################### -- WEEK {week_eval} -- TRAINING #########################")
    print()

    for epoch in range(num_epochs):

        # training phase
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

        # validiating phase
        model.eval()
        val_running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for val_features_batch, val_targets_batch in val_loader:
                val_features_batch, val_targets_batch = val_features_batch.to(device), val_targets_batch.to(device)

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

        # adding early stopping check (effort to prevent overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # saving the best model
            torch.save(model.state_dict(), MODELS_DIR / f"best_model_week{week_eval}.pth")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print()
                print(f"Early stopping triggered. Best verision saved under 'best_model_week{week_eval}.pth'")
                print()
                break