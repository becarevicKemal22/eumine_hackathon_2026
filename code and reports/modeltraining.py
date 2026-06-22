"""

The code used for training of the TakeMe2Romania_v1 model.

"""

# Paths to the training and validation datasets.
# Update these paths if the files are stored in a different folder.

train_path = "jarvis_mp_merged_v2_80.csv"
val_path = "jarvis_mp_merged_v2_20.csv"

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from pymatgen.core import Composition

from matminer.featurizers.composition import ElementProperty

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

if 'energy_above_hull_eV' in train.columns:
    train = train.drop(columns=['energy_above_hull_eV'])
if 'energy_above_hull_eV' in val.columns:
    val = val.drop(columns=['energy_above_hull_eV'])

print("Training data shape:", train.shape)
print("Validation data shape:", val.shape)

print("\nColumns in training data:")
print(train.columns.tolist())

val.head()

TARGETS = [
    "bandgap_eV",
    "formation_energy_eV_per_atom"
]

missing_train_targets = [t for t in TARGETS if t not in train.columns]
missing_val_targets = [t for t in TARGETS if t not in val.columns]

if missing_train_targets or missing_val_targets:
    raise ValueError(
        f"Missing target columns. Train: {missing_train_targets}, Validation: {missing_val_targets}"
    )

empty_rows = train[train["composition"].isna()]

print(empty_rows)

train["composition"] = train["formula"].apply(Composition)
val["composition"] = val["formula"].apply(Composition)

print("Original formula:", train.loc[0, "formula"])
print("Composition object:", train.loc[0, "composition"])

featurizer = ElementProperty.from_preset("magpie")

train_feat = featurizer.featurize_dataframe(train, col_id="composition", ignore_errors=True)

val_feat = featurizer.featurize_dataframe(val, col_id="composition", ignore_errors=True)

print("Training data after featurization:", train_feat.shape)
print("Validation data after featurization:", val_feat.shape)

print("Training data loaded after featurization:", train_feat.shape)
print("Validation data loaded after featurization:", val_feat.shape)

import numpy as np
import pandas as pd

non_feature_cols = [
    "material_id",
    "jid",
    "formula",
    "composition",
    "bandgap_eV",
    "formation_energy_eV_per_atom",
    "density_threshold",
    "exfol_threshold_meV",
    "merge_date",
    "source_db",
    "group"
]

X_train = train_feat.drop(columns=non_feature_cols)
X_val = val_feat.drop(columns=non_feature_cols)

X_train = X_train.select_dtypes(include=[np.number])

X_val = X_val[X_train.columns]

X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_train.median())

y_train = train_feat[TARGETS]
y_val = val_feat[TARGETS]

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

X_train

base_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_model)

model.fit(X_train, y_train)

print("Model training completed.")

y_pred = model.predict(X_val)

pred_df = pd.DataFrame(
    y_pred,
    columns=[f"prediction_{t}" for t in TARGETS]
)

for i, target in enumerate(TARGETS):
    mae = mean_absolute_error(y_val[target], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_val[target], y_pred[:, i]))
    r2 = r2_score(y_val[target], y_pred[:, i])

    print("\nTarget:", target)
    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2:   {r2:.4f}")

results = val[["material_id", "jid", "formula"] + TARGETS].copy()

for i, target in enumerate(TARGETS):
    results[f"prediction_{target}"] = y_pred[:, i]
    results[f"absolute_error_{target}"] = abs(
        results[target] - results[f"prediction_{target}"]
    )

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "takeme2romaniaoutput.csv"

results.to_csv(output_path, index=False)

print("Predictions saved to:", output_path)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True) 


model_save_path = output_dir / "TakeMe2Romania_v1.joblib"

joblib.dump(model, model_save_path)

print(f"Trained model saved to: {model_save_path}")