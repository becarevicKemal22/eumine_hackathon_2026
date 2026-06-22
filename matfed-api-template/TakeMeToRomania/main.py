import json
import os
from pathlib import Path
from pymatgen.core import Structure
from tqdm import tqdm

# Pretpostavljamo da se klasa nalazi u datoteci matfed_predictor.py
from matfed_predictor import BosnianPredictor


def load_structures_from_cif_folder(
    folder_path: str,
) -> tuple[list[Structure], list[str]]:
    """Učitava sve .cif datoteke iz zadanog foldera.

    Vraća listu pymatgen Structure objekata i paralelnu listu pripadajućih
    material_id-eva (naziva fajlova bez ekstenzije).
    """
    structures = []
    material_ids = []
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Greška: Direktorij {folder_path} ne postoji!")
        return [], []

    # Pronađi sve .cif datoteke (uključujući i velika slova)
    cif_files = list(folder.glob("*.cif")) + list(folder.glob("*.CIF"))
    print(f"Pronađeno {len(cif_files)} CIF datoteka. Učitavam...")

    for file_path in tqdm(cif_files, desc="Učitavanje CIF datoteka"):
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
            # Uzimamo samo ime fajla bez ekstenzije (npr. "mp-1234" od "mp-1234.cif")
            material_ids.append(file_path.stem)
        except Exception as e:
            print(f"\nGreška pri učitavanju {file_path.name}: {e}")

    return structures, material_ids


def main():
    # --- KONFIGURACIJA ---
    CIF_FOLDER = "test_input_structures"
    MODEL_PATH = None
    # ---------------------

    print("=== Inicijalizacija BosnianPredictor-a ===")
    predictor = BosnianPredictor()

    print("\n=== Učitavanje modela ===")
    predictor.load_model(model_path=MODEL_PATH)

    if predictor.model is None:
        print(
            "Greška: Model nije uspješno učitan! Provjeri putanju do .joblib fajla."
        )
        return
    print("Model je uspješno učitan.")

    print("\n=== Sakupljanje i parsiranje struktura ===")
    structures, material_ids = load_structures_from_cif_folder(CIF_FOLDER)

    if not structures:
        print("Nema struktura za predikciju. Prekidam izvršavanje.")
        return

    print(f"\n=== Pokretanje predikcije za {len(structures)} struktura ===")
    print("Ovo može potrajati jer se računaju MAGPIE deskriptori...")
    predictions = predictor.predict(structures)

    print("\n=== Predikcija završena! Usklađivanje rezultata u JSON ===")

    # Dohvaćamo informacije o timu iz deskripcije modela
    model_info = predictor.describe()

    # Kreiramo listu pojedinačnih predikcija u formatu sa slike
    json_predictions = []
    for mat_id, pred in zip(material_ids, predictions):
        json_predictions.append(
            {
                "material_id": mat_id,
                "formation_energy_per_atom": round(
                    pred["formation_energy_per_atom"], 4
                ),
                "band_gap": round(pred["band_gap"], 2),
            }
        )

    # Kreiramo finalnu strukturu JSON-a
    # Ovdje možeš promijeniti "team_name" i "model_id" ako ti trebaju specifične vrijednosti za PR
    final_output = {
        "team_name": model_info.get("team_name", "TakeMeToRomania"),
        "model_id": pred.get("model_id", "rf_magpie_bosnia_v2"),
        "matfed_api_version": "1.0",
        "predictions": json_predictions,
    }

    # Putanja za spremanje prema slici (prilagodi po potrebi)
    output_folder = Path("submissions") / final_output["team_name"]
    output_folder.mkdir(parents=True, exist_ok=True)
    output_json_path = output_folder / "predictions_test.json"

    # Spremanje u JSON fajl sa "indent=2" radi lijepe i čitljive strukture (kao na slici)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nSve predikcije su uspješno spremljene u: {output_json_path}")


if __name__ == "__main__":
    main()