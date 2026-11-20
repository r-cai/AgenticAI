# Generator schema-recovery fix — 2026-11-19

## Reported failure

Both uploaded `proposal.json` files contained the same exception:
`candidates.1.composition: ensembles require 2-3 model branches`.
A single invalid `voting` composition caused `GeneratorOutput.model_validate_json` to reject the **entire** response, including any valid proposals, for two iterations.

## Changes

- `agents/deep.py`: parse the structured response, validate **each** candidate independently, retain all valid proposals, and request only the missing candidates in **one targeted retry**. Invalid structures are rejected, never silently repaired. The LLM now receives a concrete two-branch voting example and explicit constraints. When all proposals fail, a `GeneratorValidationError` carries structured rejection metadata.
- `agents/schemas.py`: use a private Pydantic attribute `_validation_errors` to carry generator-side diagnostics without modifying the LLM's JSON schema.
- `orchestration/evidence_integration.py`: propagate rejected LLM candidate details into the iteration audit (`evidence_translation_audit.json`); preserve valid proposals if the retry fails. No change to the configured candidate budget or research composition slots.
- `tests/test_generator_recovery.py`: mock Ollama responses reproducing the original error and test partial recovery, bounded retry, and all-invalid failure.

## Instructions

Extract this ZIP as a **new folder** rather than overwriting your earlier experiments; keep previous reports for reproducibility. Install `requirements.txt`, then run:

```bash
python -m unittest discover -s tests -v
python main.py --csv data/data-2.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/B_rag_context_only.yaml
```

Only after B successfully evaluates non-baseline candidates, run C using `configs/C_rag_evidence_compiler.yaml`. Keep candidate budget, seed, dataset, and evaluation rules matched. This fix has been tested offline with mocked Ollama replies; live Ollama behavior still needs verification.

## Scientific caveat

The fix improves generator robustness, **not** predictive accuracy and **not** measured RAG effectiveness. Any improvement must be measured in new controlled B/C runs. The system may still fail to produce proposals if Ollama repeatedly emits invalid configurations, but the cause and rejected candidates will be auditable.
