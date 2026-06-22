import json
import os
from pathlib import Path
from pymatgen.core import Structure
from tqdm import tqdm

from matfed_predictor import TakeMe2RomaniaPredictor


def load_structures_from_cif_folder(
    folder_path: str,
) -> tuple[list[Structure], list[str]]:
    
    structures = []
    material_ids = []
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder {folder_path} does not exist!")
        return [], []

    cif_files = list(folder.glob("*.cif")) + list(folder.glob("*.CIF"))
    print(f"Found {len(cif_files)} CIF files. Loading...")

    for file_path in tqdm(cif_files, desc="Loading CIF files.."):
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
           
            material_ids.append(file_path.stem)
        except Exception as e:
            print(f"\nError while loading {file_path.name}: {e}")

    return structures, material_ids


def main():

    CIF_FOLDER = "test_input_structures"
    MODEL_PATH = "model/TakeMe2Romania_v1.joblib"
   
    print("=== Initialising TakeMe2RomaniaPredictor ===")
    predictor = TakeMe2RomaniaPredictor()

    print("\n=== Loading model ===")
    predictor.load_model(model_path=MODEL_PATH)

    if predictor.model is None:
        print(
            "Error: Model not found!"
        )
        return
    print("Model loaded properly.")

    print("\n=== Structure parsing... ===")
    structures, material_ids = load_structures_from_cif_folder(CIF_FOLDER)

    if not structures:
        print("No structures to parse. Exiting...")
        return

    print(f"\n=== Running predictions for {len(structures)} structures ===")
    print("This can take a while...")
    predictions = predictor.predict(structures)

    print("\n=== Predictions made! Exporting to JSON. ===")

  
    model_info = predictor.describe()

   
    json_predictions = []
    for mat_id, pred in zip(material_ids, predictions):
        json_predictions.append(
            {
                "material_id": mat_id,
                "formation_energy_per_atom":
                    pred["formation_energy_per_atom"],
                "band_gap": pred["band_gap"],
            }
        )

  
    final_output = {
        "team_name": model_info.get("team_name", "TakeMe2Romania"),
        "model_id": model_info.get("model_id", "TakeMe2Romania_v1"),
        "matfed_api_version": "1.0",
        "predictions": json_predictions,
    }

    output_json_path = Path("predictions_test.json")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions sucessfully saved as: {output_json_path}")


if __name__ == "__main__":
    main()