# Experiment D: CBIS-DDSM zero eligible evidence — targeted recovery

## Why D stopped

Your console log says **12 extracted records, 0 eligible evidence**. Strict D is
not permitted to cite an arbitrary paper; the excerpt must describe a registered
feature-selection / transformation / ensemble operation compatible with tabular
modeling. D previously retried Qwen three times without evidence, so only the
Random Forest baseline was evaluated.

## Changes in this package

1. `rag/evidence_compiler.py`: the image-only filter no longer considers the
   word **mammography** or **radiology** alone proof of pixel-level modeling;
   explicitly image-based methods (CNN, U-Net, MRI, image segmentation, etc.)
   remain excluded.
2. `rag/evidence.py`: the extraction prompt now asks for a verbatim excerpt
   containing the method name, when a method is actually described.
3. `rag/evidence_recovery.py` (NEW): if strict D has no eligible records, perform
   at most 2 targeted retrieval rounds using tabular classification, feature
   selection and ensemble queries. Search different queries separately so the
   first irrelevant paper set cannot exhaust the retrieval budget. Re-extract
   claims with the same LLM and apply the SAME source-grounding, caution, and
   composition validation policy. Do not invent papers or modeling methods.
4. `orchestration/controller.py`: recover BEFORE entering the generator loop,
   save `evidence_recovery_audit.json`, keep `research.json`, `evidence.json`,
   and `eligible_evidence.json` up to date and record additional RAG rounds.
   If evidence remains unusable, skip invalid D generation attempts and report
   `stop_reason: insufficient_eligible_evidence`, `strict_D_completed: false`.
   The baseline CV is still included for diagnosis but MUST NOT be scored as D.
5. `configs/D_rag_llm_grounded.yaml`: recovery is enabled for D only.
6. `tests/test_evidence_recovery_D.py`: additional regression tests.

## Run (from the directory that contains main.py)

```cmd
python -m unittest discover -s tests -v
python main.py --csv data/data-cbis-ddsm.csv --target diagnosis --task "Predict whether a patient has at least one malignant breast lesion using CBIS-DDSM mammographic metadata" --config configs/D_rag_llm_grounded.yaml
```

Output directory is printed as an absolute path, and contains:

- `research.json`, `evidence.json`, `eligible_evidence.json`
- `evidence_selection_audit.json`, `evidence_recovery_audit.json`
- `final_report.json` and `level2_experiment_report.pdf`

## Interpretation

Successful D means `strict_D_completed=true` **and** at least one candidate
with `origin=llm_evidence_design` was evaluated. A high baseline-only score
when `strict_D_completed=false` is **not** an Experiment D result.

For scientific B/C/D comparison, reuse the **same saved literature** after
recovery via `rag.replay_from_run: experiments/runs/<RUN_ID>`. Replay mode
never performs recovery or changes the corpus. The number of distinct evaluated
candidates, rejection count, time, and actual grounding quality must still be
compared. Do not choose models based on repeatedly inspected holdout scores.

### If zero eligible evidence persists

Inspect `evidence_recovery_audit.json` for `retrieval_errors`, `new_papers`,
`new_evidence`, and per-reason rejection counts; also inspect
`evidence_selection_audit.json`. A network outage or inaccessible API can
prevent recovery. This must **not** be silently converted into ordinary B.

For a quick diagnostic, temporarily increase `rag.top_k`, `evidence.max_papers`
and `evidence_selection.max_records` together to 10, but never disable
`require_structural` or source-excerpt grounding just to force an apparent D.

## Tested

Offline automated unit/integration tests pass. Live Ollama + online retrieval
is environment-dependent and has not been validated from this workspace.
