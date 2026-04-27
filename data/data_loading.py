import pandas as pd
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from data_cleaning import rotate_direction_and_orientation, make_plays_left_to_right, calculate_velocity_components, pass_attempt_merging, label_offense_defense_multitask, select_augmented_frames, data_augmentation, prepare_frame_multitask_data

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TENSORS_DIR = Path(__file__).resolve().parent.parent / "tensors"
TENSORS_DIR.mkdir(parents=True, exist_ok=True)

def process_week_data(week_number, plays):
  file_path = DATA_ROOT / f"week{week_number}.csv"
  week = pd.read_csv(file_path)
  print(f"Finished reading Week {week_number} data")

  # Data cleaning functions
  week = rotate_direction_and_orientation(week)
  week = make_plays_left_to_right(week)
  week = calculate_velocity_components(week)
  week = pass_attempt_merging(week, plays)
  week = label_offense_defense_multitask(week, plays)

  week['week'] = week_number
  week['uniqueId'] = week['gameId'].astype(str) + "_" + week['playId'].astype(str)
  week['frameUniqueId'] = (
      week['gameId'].astype(str) + "_" +
      week['playId'].astype(str) + "_" +
      week['frameId'].astype(str))

  snap_frames = week[week['event'] == 'ball_snap'].groupby('uniqueId')['frameId'].first()
  week = week.merge(snap_frames.rename('snap_frame'), on='uniqueId', how='left')
  week['frames_from_snap'] = week['frameId'] - week['snap_frame']
  week = week[week['frameId'] % 2 == 0]
  week = week[(week['frames_from_snap'] >= -150) & (week['frames_from_snap'] <= 50)]

  num_unique_frames = len(set(week['frameUniqueId']))
  selected_frames = select_augmented_frames(week, int(num_unique_frames / 3), sigma=5)
  week_aug = data_augmentation(week, selected_frames)

  week = pd.concat([week, week_aug])

  print(f"Finished processing Week {week_number} data")
  print()

  return week


# read raw data
games = pd.read_csv(DATA_ROOT / "games.csv")
players = pd.read_csv(DATA_ROOT / "players.csv")
plays = pd.read_csv(DATA_ROOT / "plays.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_weeks = []

for week_number in range(1, 9):
  week_data = process_week_data(week_number, plays)
  all_weeks.append(week_data)

all_tracking = pd.concat(all_weeks, ignore_index=True)
all_tracking = all_tracking[(all_tracking['team'] != 'football') & (all_tracking['passAttempt'] == 1)]


features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
target_columns = ["pff_passCoverage", "pff_passCoverageType"]

# looping through weeks & saving each week's training data + validating data
for week_eval in range(1, 9):

  train_df = all_tracking[all_tracking['week'] != week_eval]
  val_df = all_tracking[all_tracking['week'] == week_eval]

  train_df = train_df[['frameUniqueId', 'frameId', 'event', 'x_clean', 'y_clean', 'v_x', 'v_y', 'defensiveTeam', 'pff_passCoverage', 'pff_passCoverageType', 'defense']]
  val_df = val_df[['frameUniqueId', 'frameId', 'event', 'x_clean', 'y_clean', 'v_x', 'v_y', 'defensiveTeam', 'pff_passCoverage', 'pff_passCoverageType', 'defense']]

  train_features, train_targets = prepare_frame_multitask_data(train_df, features, target_columns)
  val_features, val_targets = prepare_frame_multitask_data(val_df, features, target_columns)

  train_dataset = TensorDataset(train_features, train_targets['pff_passCoverage'], train_targets['pff_passCoverageType'])
  val_dataset = TensorDataset(val_features, val_targets['pff_passCoverage'], val_targets['pff_passCoverageType'])

  print(f"Week {week_eval} Tensor: {train_features.shape}") # should be: torch.Size([X, 22, 5]) where X is num frames, 22 is num plays, 5 is num features (as definied above)
  print(f"Week {week_eval} Indiv Check: {train_features[63][0]}") # should be: tensor([x_cord, y_cord, v_x,  v_y,  1/0])

  torch.save(train_features, TENSORS_DIR / f"features_training_week{week_eval}preds.pt")
  torch.save(val_features, TENSORS_DIR / f"features_val_week{week_eval}preds.pt")

  torch.save(train_targets['pff_passCoverage'], TENSORS_DIR / f"targets_training_coverage_week{week_eval}preds.pt")
  torch.save(val_targets['pff_passCoverage'], TENSORS_DIR / f"targets_val_coverage_week{week_eval}preds.pt")
  
  torch.save(train_targets['pff_passCoverageType'], TENSORS_DIR / f"targets_training_manzone_week{week_eval}preds.pt")
  torch.save(val_targets['pff_passCoverageType'], TENSORS_DIR / f"targets_val_manzone_week{week_eval}preds.pt")