from typing import Any, Dict, List

from jsonschema import validate


PREDICTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": [
            "formation_energy_per_atom",
            "band_gap",
            "model_id",
            "data_sources_used",
        ],
        "properties": {
            "formation_energy_per_atom": {"type": "number"},
            "band_gap": {"type": "number"},
            "model_id": {"type": "string"},
            "data_sources_used": {
                "type": "array",
                "items": {"type": "string"},
            },
            "uncertainty": {"type": "number"},
        },
        "additionalProperties": True,
    },
}


def validate_predictions(predictions: List[Dict[str, Any]]) -> None:
    validate(instance=predictions, schema=PREDICTION_SCHEMA)
