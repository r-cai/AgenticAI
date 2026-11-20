# Dataset- and experiment-specific final report naming

The report writer uses the `--csv` filename and the `--config` filename.
Run the commands from the directory containing `main.py`.

## New runs

A run like:

```powershell
python main.py --csv data/data-3.csv --target diagnosis --task "Predict malignant versus benign breast tumors using WDBC measurements" --config configs/A_no_rag.yaml
```

writes **four JSON copies**:

- `experiments/runs/<run_id>/final_report.json` (original path, unchanged API)
- `experiments/runs/<run_id>/final_report_data-3_A.json` (named copy for the run)
- `experiments/reports/final_report_data-3_A_<run_id>.json` (archived copy, never intentionally overwritten)
- `experiments/reports/final_report_data-3_A.json` (latest convenient copy, overwritten by the next A run on data-3)

The analogous names for the second dataset are:

- `final_report_data-cbis-ddsm_A.json`
- `final_report_data-cbis-ddsm_B.json`
- `final_report_data-cbis-ddsm_C.json`
- `final_report_data-cbis-ddsm_D.json`

The run ID is also embedded in the archive filename so reruns never erase previous outputs.
The PDF report remains in the original run directory, with its existing name; the JSON files contain the PDF path when generation succeeds.

The experiment label comes from the selected YAML filename (`A_...`, `B_...`, `C_...`, `D_rag_llm_grounded.yaml`). The legacy `D_no_rag_level2.yaml` is **not** mislabeled as the evidence-grounded Experiment D.
If you use a custom YAML filename, its stem is used rather than guessing the experiment arm.

## Existing reports (no rerun required)

After extracting the updated project over your existing project or copying your old `experiments/runs` directory into it:

```powershell
python -m scripts.backfill_named_reports
```

This reads `config_snapshot.json` and `final_report.json` in every run, identifies A/B/C/D from the saved settings, and creates the named copies under `experiments/reports/`. **It does not modify the original JSON**. It skips runs that lack either file.

You can specify custom locations:

```powershell
python -m scripts.backfill_named_reports --runs-dir "C:\\your-old-project\\experiments\\runs" --reports-dir "experiments/reports"
```

When using `--runs-dir` pointing to an old project, the per-run named copy is written beside the old report, whereas the archived and latest copies go to `--reports-dir`.

## Experiment D and evidence eligibility

This package retains the previous bounded, strict literature-recovery behavior for **both** `data-3.csv` and `data-cbis-ddsm.csv`. It never fabricates a paper or relaxes citation requirements merely to fill candidate slots. If no eligible evidence is obtained, the run report will still be saved with the D filename, but it will have `strict_D_completed: false` and `stop_reason: insufficient_eligible_evidence` so it is not mistaken for a successful D experiment. Inspect `evidence_recovery_audit.json` and `evidence_selection_audit.json` in the run folder for causes.

**For fair A/B/C/D research comparisons:** continue to use matched valid candidate-evaluation counts and the same frozen evidence corpus for B/C/D, where applicable. File naming makes results easier to organize; it does not itself make experiments statistically comparable.
