# MatFed API Template (EuMINe DataBridge Hackathon 2026)

This repository defines the **MatFed API v1** contract used in the EuMINe DataBridge Hackathon.
Every team submission must implement this interface so organizer tooling can load models,
run predictions consistently, and validate output format automatically.

## What this repository provides

- A strict abstract base class: `MatFedPredictor`
- A JSON schema validator for predictions
- A working example predictor (`RandomForestPredictor`)
- A compliance test suite (`pytest`) used by organizers

## Repository layout

```text
matfed-api-template/
├── README.md
├── requirements.txt
├── matfed_api/
│   ├── __init__.py
│   ├── predictor.py            # abstract interface (MatFedPredictor)
│   └── schema.py               # output JSON schema + validator
├── example_implementation/
│   ├── my_predictor.py         # reference implementation
│   └── run_example.py          # local demo runner
└── tests/
        ├── conftest.py             # dynamic predictor loader via MY_PREDICTOR
        ├── test_interface.py       # 5 compliance checks
        └── sample_structures/
                ├── test_001.cif
                ├── test_002.cif
                ├── test_003.cif
                ├── test_004.cif
                └── test_005.cif
```

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

If you use conda/venv, activate it first.

## Quick start

Run interface compliance tests against the built-in example implementation:

```bash
pytest tests/test_interface.py -v
```

Expected result: all tests pass.

## API contract (MatFed API v1)

Your predictor class must subclass `MatFedPredictor` and implement all methods:

```python
load_model(self, model_path: str) -> None
predict(self, structures: List[Structure]) -> List[Dict]
describe(self) -> Dict
```

### `load_model(...)`

- Loads model artifacts from disk (weights, scalers, configs, etc.)
- Must not return anything

### `predict(...)`

- Input: list of `pymatgen.core.Structure`
- Output: list of dictionaries, same length as input
- Each output dict must contain at least:

    - `formation_energy_per_atom` (float, eV/atom)
    - `band_gap` (float, eV)
    - `model_id` (str)
    - `data_sources_used` (list[str])

Optional fields are allowed (for example `uncertainty`) and are not rejected.

### `describe(...)`

Must return metadata dictionary with at least:

- `team_name`
- `model_type`
- `api_version`
- `data_sources`

## Validate output programmatically

You can validate predictions before submission:

```python
from matfed_api import validate_predictions

validate_predictions(predictions)
```

If the payload is invalid, an exception is raised.

## Test your own predictor class

The test fixture loads a class from `MY_PREDICTOR` in the format:

```text
package.module.ClassName
```

Example:

```bash
export MY_PREDICTOR=mypkg.my_model.MyPredictor
pytest tests/test_interface.py -v
```

If `MY_PREDICTOR` is not set, tests default to:

```text
example_implementation.my_predictor.RandomForestPredictor
```

## Run the bundled example

```bash
python example_implementation/run_example.py
```

This loads sample CIF files, runs predictions, and prints model metadata.

## Common issues

### `ModuleNotFoundError` when setting `MY_PREDICTOR`

Cause: import path is wrong or package is not installed in current environment.

Fix:

- verify module path
- install your package (`pip install -e .` if local)
- rerun tests

### Abstract class instantiation error

If you see `TypeError` on predictor creation, your class is missing one or more required methods.

### Schema validation failure

Check key names and types exactly (`band_gap` not `bandgap`, list[str] for `data_sources_used`, etc.).

## Submission checklist

Before final submission, confirm:

- interface tests pass (`pytest tests/test_interface.py -v`)
- `predict()` returns one item per input structure
- required keys are present and numeric fields are valid floats
- `describe()` includes required metadata fields
- dependencies are documented in `requirements.txt` or equivalent

## Notes for participants

- CPU-compatible execution is recommended for reproducibility.
- Keep data-source provenance explicit in `data_sources_used`.
- A simple, robust model that passes all checks is preferable to a complex
    model with fragile runtime assumptions.
