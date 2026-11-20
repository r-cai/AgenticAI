# Experiment D — Guided Literature-Backed Composition Fix

## Problem reproduced from the CBIS-DDSM console log

- Initial RAG: 5 papers, 6 extracted claims, **0 eligible** structural claims.
- Targeted recovery: 1 additional paper and one eligible methodological claim.
- Original `qwen3.5:4b` was asked to generate complete nested model graphs; it repeatedly returned malformed branches, standalone models, missing evidence IDs, or architectures that had no supporting method excerpt.
- The controller consequently ran only the Random Forest baseline (CV F1 ~0.6948 in the shared log).
- This is a proposal validity / evidence scarcity problem, **not a negative predictive result for grounded D**.

## Fix: concise guided design (NOT template compiler C)

The **same Qwen generator** now proposes small, typed **scientific design choices**:

```json
{
  "choices": [
    {
      "evidence_id": "<verified-id-from-eligible-evidence>",
      "operation": "rfe",
      "base_models": ["logistic_regression"],
      "n_features": 12,
      "lr_c": 1.0,
      "rf_trees": 150,
      "rf_depth": 8,
      "voting": "soft",
      "hypothesis": "Test whether feature elimination reduces redundant measurements."
    }
  ]
}
```

The deterministic compiler builds an approved Level 2 pipeline, checks the source excerpt truly supports the **structural operation**, validates the candidate, and submits it to the original trainer and CV. The LLM still chooses the method (from actual retrieved evidence), base models, feature count, and hyperparameters. Unlike C, it is **not** a fixed template per paper. It does **not** synthesize Python or claim the paper supported the unquoted implementation details.

The compact response format is intended to improve compatibility with `qwen3.5:4b`. It is **not** a guarantee that the model always complies. Invalid choices are rejected and logged; no evidence or models are fabricated.

The existing free-form D generator is retained for research comparison.

## Settings

`configs/D_rag_llm_grounded.yaml` now contains:

```yaml
llm_evidence_design:
  enabled: true
  strategy: guided      # switch to freeform to run the previous D behavior
  validation_retries: 2
  empty_evidence_recovery:
    enabled: true
    max_rounds: 2
    top_k: 10
    extract_max_papers: 10
    target_method_groups: 2  # attempt a second method; limited to two rounds
```

Shorter recovery queries are used for arXiv in both rounds, because query combinations with too many `AND` terms had no results. A Semantic Scholar 429 still remains possible. If recovery cannot find a usable paper, Experiment D **fails closed** and reports an incomplete experiment.

## Commands (from the folder containing `main.py`)

```powershell
python -m unittest discover -s tests -v
python main.py --csv data/data-cbis-ddsm.csv --target diagnosis --task "Predict whether a patient has at least one malignant breast lesion using CBIS-DDSM mammographic metadata" --config configs/D_rag_llm_grounded.yaml
```

For Wisconsin, copy `data-3.csv` into `data/` and run:

```powershell
python main.py --csv data/data-3.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/D_rag_llm_grounded.yaml
```

Each run writes a unique `experiments/runs/<run-id>/final_report.json` and a convenience copy `experiments/reports/final_report_<dataset>_D.json` with a run-ID archive beside it.

## What to verify in the report

- `strict_D_completed` is `true` **only if at least one validated D candidate completed CV**.
- `llm_grounded_strategy` is `guided`.
- `research_influence.llm_grounded_candidates_executed` is positive.
- `evidence_selection_audit` and `evidence_recovery_audit` provide provenance diagnostics.
- `evidence_translation_audit` contains accepted / rejected designs and reasons.
- `candidate.config.composition` shows the actual composition trained.

## Scientific caveats

The new D **changes the generator interface**, so old D scores cannot be treated as direct algorithmic comparisons with new D. Re-run A/B/C/D on identical dataset/CV settings, comparable numbers of actual evaluations, and frozen literature where possible. A no-RAG run cannot legitimately use recovered scientific citations. Do not use repeatedly inspected holdout scores to choose a research method. CBIS patient-aggregated metadata are a different diagnostic task from the WDBC cell-nucleus features; examine potentially label-proxy variables and prefer independent evaluation for final claims.

Offline verification includes original and guided module tests, actual scikit-learn CV, and a mocked end-to-end controller test. Live Ollama + online retrieval still need verification in the user's Windows environment.
