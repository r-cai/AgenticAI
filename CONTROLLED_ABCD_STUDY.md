# Controlled, repeated-seed A/B/C/D experimental protocol

This package extends the **JSON Literature Knowledge Base** pipeline without discarding its original A/B/C/D configs or the restored paper JSON files. The four previously uploaded CBIS-DDSM reports are preserved under `examples/pilot_cbis_20261119` and are **not** combined with controlled runs.

## Research question

Does scientific literature improve an autonomous ML agent's ability to propose and evaluate model compositions? Analyze separately (1) CV/holdout predictive performance, (2) whether evidence affects executable model construction, (3) evidence-to-operation faithfulness, (4) candidate-generation reliability and wall time.

## Pre-registered defaults

- Arms: **A** no RAG, **B** RAG context, **C** deterministic evidence→composition compiler, **D** LLM-guided, strictly evidence-grounded composition.
- Seeds: **42, 52, 62, 72, 82**; 5 paired 80/20 stratified train/holdout splits. **All four arms share exactly the same split within each seed.** Seeds give *different*, overlapping train/holdout partitions.
- Selection: repeated stratified **5-fold × 3-repeat CV on training data only**, Macro F1, then fit the CV-selected best model on the full training split and score the holdout once. Baseline Random Forest is identical in every arm and is excluded from the *generated candidate budget*.
- Generated candidates: **six successfully cross-validated candidates per arm per seed**. If more generation is needed to obtain six valid, nonduplicate designs, the agent may run up to nine iterations (three slots per iteration; up to five generator call rounds per iteration). There is **no automatic replacement with ungrounded models** in D. If six cannot be evaluated, the run is marked incomplete and excluded from strictly matched comparisons.
- Candidate *evaluations* are matched, but **LLM calls, amount of search effort, and wall time are not forced to be equal**. Record those as separately evaluated costs; this design isolates accuracy at a fixed number of evaluated candidates, not a fixed token or compute budget.
- The critic is disabled in every arm. LLM generator and evidence extraction settings otherwise come from the baseline configs. No online retrieval occurs in B/C/D: all read the **same frozen research.json and evidence.json** copied to the suite snapshot before any run. A has no RAG.
- The knowledge base stored at `C:\Users\charl\anaconda3\AgenticAI\RAG_Knowledge_Base` is retained; the controlled B/C/D runs **do not modify or refresh the frozen literature**.
- Original dataset file bytes, target, task, frozen literature, seed list, and budget are SHA256-checked in the suite manifest. The same experiment task is used across A/B/C/D in each dataset.

## Commands (PowerShell, from the project directory containing main.py)

```powershell
$env:AGENTIC_RAG_KB_DIR = "C:\Users\charl\anaconda3\AgenticAI\RAG_Knowledge_Base"
python -m unittest discover -s tests -v

# CBIS-DDSM: five matched seeds and six evaluated candidates per arm
python -m scripts.run_controlled_abcd --dataset cbis --seeds 42 52 62 72 82 --budget 6 --max-iterations 9 --suite-id cbis_repeated_v1
python -m scripts.analyze_controlled_abcd --suite experiments/controlled/cbis_repeated_v1 --bootstrap 2000

# WDBC: same protocol, with its own frozen WDBC literature snapshot
python -m scripts.run_controlled_abcd --dataset wdbc --seeds 42 52 62 72 82 --budget 6 --max-iterations 9 --suite-id wdbc_repeated_v1
python -m scripts.analyze_controlled_abcd --suite experiments/controlled/wdbc_repeated_v1 --bootstrap 2000
```

For a short protocol check, first run `--seeds 42 52 --suite-id cbis_smoke_v1 --dry-run` (no Ollama calls), and later remove `--dry-run` while keeping the same suite ID and parameters. The runner is resumable: it skips completed arm reports whose prediction files exist. Use `--force` only to deliberately replace a suite's per-arm aliases; the original run directories and unique archived reports remain.

## Outputs

```text
experiments/controlled/cbis_repeated_v1/
  suite_manifest.json                        # study design, dataset SHA, frozen SHA
  frozen_corpus/
    research.json                            # frozen once per suite
    evidence.json
  configs/<seed>/<A|B|C|D>_*.yaml           # config snapshots
  logs/seed_<seed>_<arm>.log                # full text trace
  reports/seed_<seed>_<arm>.json            # one report per experiment and seed
  predictions/seed_<seed>_<arm>_selected.csv
  predictions/seed_<seed>_<arm>_baseline.csv
  analysis/
    per_run_metrics.csv
    paired_per_seed_bootstrap.csv
    paired_across_seeds_descriptive.csv
    quality_audit.csv
    analysis_summary.json
```

Every run also saves its own `experiments/runs/<run_id>/split_manifest.json` (actual row indices, dataset/hash, split/hash), `holdout_predictions.csv` (row index, true class, predicted class, available predicted class probabilities, dataset/split SHA, seed, candidate ID), `baseline_holdout_predictions.csv`, and its original and named JSON reports. **No holdout predictions are supplied to the generator or used in model selection.**

`quality_audit.csv` records which arms/seeds are excluded from paired comparisons, especially **unfilled budgets**, **missing predictions**, **wrong corpus/dataset hashes**, and **different splits**. `strict_D_completed` is `null` in A/B/C rather than misleadingly `true`.

## Paired differences and uncertainty

For each fully matched seed, pair predictions by the original CSV **row_index**, and verify ground-truth labels, dataset SHA, split SHA, and frozen evidence digest. Compare **all six A/B/C/D pairs** using Macro F1 and accuracy, plus ROC-AUC when *both* models output probabilities for the malignant class `M`.

Use **2,000 stratified paired bootstrap resamples of holdout patients within each seed** for approximate 95% percentile intervals of each difference. Each bootstrap iteration resamples **the same patient rows for both arms** so dependence is preserved. Intervals are **conditional on the split and the already-selected models**; they omit uncertainty from training/search/model selection. Do **not** treat CV folds, bootstrap replicates, or the overlapping holdouts across seeds as independent biological samples. Cross-seed summaries report means, standard deviations, ranges, and positive-difference counts only, **not** independent-sample significance tests.

For stronger confirmatory results, plan an untouched external test cohort or nested outer evaluation with non-overlapping patient groups, retain per-patient predictions, and avoid repeated inspection of any one holdout to redesign the agent.

## Research integrity

The default CBIS frozen corpus is based on restored historical literature and contains some methods previously studied on **WDBC rather than CBIS-DDSM**. The supported operation can motivate an experimental hypothesis but does **not** certify its transfer performance or validate the LLM's implementation details/hyperparameters. Document for each D candidate: citation, source dataset/modality, verbatim method excerpt, operation-level match, proposed implementation, CV performance, and observed holdout outcomes. In particular, **WPFS feature selection is not interchangeable with SelectKBest**; mark such adaptations as *methodologically inspired*, not exact replications, and consider tightening the source-to-operation validator before final publication.

## Limitations of offline testing

Local automated tests exercise the runner configuration, split alignment, strict candidate budgets, validation checks, prediction CSV writing, and paired-bootstrap analysis. **The full live repeated-seed study requires Ollama/Qwen on your Windows computer and is not included as a fabricated result.** If an arm cannot fill the six-candidate budget, that seed is omitted from the paired analysis rather than falsely treated as balanced.
