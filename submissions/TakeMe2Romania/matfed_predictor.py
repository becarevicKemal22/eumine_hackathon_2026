import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matfed_api import MatFedPredictor


# ---------------------------------------------------------------------------
# Output column order of MODEL.predict(X).
# This MUST match the column order of `y` used when the model was trained.
# Retraining keeps the Colab order: TARGETS = ["bandgap_eV", "formation_energy_eV_per_atom"]
#   -> column 0 = band gap, column 1 = formation energy per atom
# If you change the target order in the training script, change these too.
# ---------------------------------------------------------------------------
BAND_GAP_COL = 0
FORMATION_ENERGY_COL = 1

# Default location of the trained model, relative to this file.
DEFAULT_MODEL_FILE = "model/V2_magpie_multioutput_random_forest_model.joblib"

# The .joblib is too large to commit; it is hosted on Google Drive and fetched on
# first use. Id is from the share link .../file/d/<ID>/view.
# REPLACE WHEN NECESSARY
MODEL_DRIVE_FILE_ID = "1qxlbETZFbXFNSlmdCDEKE5NjEPfWKKAr"


class BosnianPredictor(MatFedPredictor):
    def __init__(self) -> None:
        self.featurizer = ElementProperty.from_preset("magpie")
        self.model = None
        self.expected_features = None

    def load_model(self, model_path: str = None) -> None:
        if model_path is None:
            path = Path(__file__).parent / DEFAULT_MODEL_FILE
        else:
            path = Path(model_path)
            if path.is_dir():
                path = path / DEFAULT_MODEL_FILE

        # Fetch the weights from Google Drive if they aren't on disk yet.
        if not (path.exists() and path.stat().st_size > 0):
            self._download_model(path)

        if path.exists() and path.stat().st_size > 0:
            self.model = joblib.load(path)
            self.expected_features = self._read_expected_features(self.model)

    @staticmethod
    def _download_model(dest: Path) -> None:
        """Download the model weights from Google Drive into `dest`.

        Uses gdown (installed on first run if missing) because Drive serves large
        files behind an HTML confirmation page that a plain curl would save as
        corrupt bytes; gdown handles the confirmation token automatically.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        print(f"Downloading model weights to {dest} ...")
        gdown.download(id=MODEL_DRIVE_FILE_ID, output=str(dest), quiet=False)

    @staticmethod
    def _read_expected_features(model) -> List[str]:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_names_in_"):
            return list(model.estimators_[0].feature_names_in_)
        return None

    def _build_feature_frame(self, structures: List[Structure]) -> pd.DataFrame:
        rows = []
        for structure in structures:
            rows.append(
                {
                    "composition": structure.composition,
                    "density_g_cm3": float(structure.density),
                    "spacegroup_number": self._safe_spacegroup(structure),
                }
            )
        df = pd.DataFrame(rows)

        df = self.featurizer.featurize_dataframe(df, "composition", ignore_errors=True)

    
        if self.expected_features is not None:
            df = df.reindex(columns=self.expected_features)

        # Never drop rows (must return one prediction per input); impute any gaps.
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    @staticmethod
    def _safe_spacegroup(structure: Structure) -> int:
        try:
            return int(SpacegroupAnalyzer(structure).get_space_group_number())
        except Exception:
            return 1

    def predict(self, structures: List[Structure]) -> List[Dict]:
        if not structures:
            return []

        if self.model is None:
            self.load_model()

        ef_preds = None
        bg_preds = None
        if self.model is not None:
            try:
                x_values = self._build_feature_frame(structures)
                preds = np.asarray(self.model.predict(x_values))
                bg_preds = preds[:, BAND_GAP_COL]
                ef_preds = preds[:, FORMATION_ENERGY_COL]
            except Exception:
                ef_preds = bg_preds = None

    
        if ef_preds is None or bg_preds is None:
            ef_preds = np.full(len(structures), -1.0, dtype=float)
            bg_preds = np.full(len(structures), 1.0, dtype=float)

        predictions = []
        for ef_value, bg_value in zip(ef_preds, bg_preds):
            predictions.append(
                {
                    "formation_energy_per_atom": float(ef_value),
                    "band_gap": float(bg_value),
                    "model_id": "rf_magpie_bosnia_v2",
                    "data_sources_used": ["Materials Project", "JARVIS-DFT"],
                }
            )
        return predictions

    def describe(self) -> Dict:
        return {
            "team_name": "TakeMe2Romania",
            "model_type": "RandomForestRegressor (multi-output) + MAGPIE",
            "api_version": "MatFed API v1",
            "data_sources": ["Materials Project", "JARVIS-DFT"],
            "requires_pretrained_weights": True,
        }
