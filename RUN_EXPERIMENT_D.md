# Experiment D: LLM-guided, evidence-grounded Level-2 compositions

This upgrade extends the corrected Level-2 package. It preserves Experiment B
(RAG as LLM context) and Experiment C (template-based evidence compilation), and
adds **a new experiment, D**. The preexisting `D_no_rag_level2.yaml` is NOT this
new research treatment: it is a no-RAG composition control.

## What makes D different?

- **B:** Qwen proposes arbitrary valid configurations and may use papers as context.
- **C:** One slot (when available) is filled by a deterministic template translating a source excerpt to a known composition; remaining slots come from Qwen.
- **D:** Qwen designs all proposals but each must be a Level-2 composition that cites a retrieved `evidence_id` and **executes a structural operation mentioned in the cited excerpt**. The deterministic sklearn compiler builds and validates the actual pipeline. D does NOT use C's fixed template compositions.

D rejects single-estimator proposals, fake citations, papers with irrelevant imaging modalities, and citations that do not support a structural operation actually present in the proposal. A conservative caution-language guard rejects explicit negative evidence as positive support. The mapping is audited as *operation-level* evidence; hyperparameters and unrelated branches are LLM-selected hypotheses, **not** representations of settings used by the paper.

D's strict filtering can produce **fewer** accepted candidates than the configured maximum. Monitor successful CV evaluations and LLM calls, and do not interpret the empty slots as performance.

## One-command live D trial

From the extracted package root (where `main.py` lives), install once:

```bash
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b
ollama pull qwen3:4b
python -m unittest discover -s tests -v
```

Copy the WDBC CSV to `data/data-2.csv` or supply its path, then:

```bash
python main.py --csv data/data-2.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/D_rag_llm_grounded.yaml
```

For Windows PowerShell, the above single-line command works as-is. The config:

```yaml
rag:
  enabled: true
  replay_from_run: null   # live retrieval, unless set as below

evidence:
  enabled: true
  mode: llm

evidence_composition:
  enabled: false          # C's fixed templates disabled

llm_evidence_design:
  enabled: true           # D's LLM-guided design enabled
  validation_retries: 2
```

The rest of `configs/D_rag_llm_grounded.yaml` matches B/C: seed 42,
20% holdout, 5-fold × 3 repeated CV, macro F1, 3 iterations,
3 candidates per iteration, critic disabled, `id` excluded.

## Preferred research comparison: freeze the literature first

Live retrieval may return different papers in B, C and D. To reduce this
confounding, choose a **previous successful B/C run directory** containing
`research.json` and `evidence.json` (for example,
`experiments/runs/20261119_154054_744841/`, if present on your computer).

In each of these three configs:

- `configs/B_rag_replay.yaml`
- `configs/C_rag_replay.yaml`
- `configs/D_rag_llm_grounded.yaml`

set:

```yaml
rag:
  enabled: true
  replay_from_run: "experiments/runs/20261119_154054_744841"
```

Use the **same actual path** in all three configs. During these runs the
literature and extracted evidence are loaded from disk; they are not retrieved
or re-extracted and no follow-up research takes place. Each report records
`research_source: frozen_replay` and `research_evidence_sha256` for verification.

Run B, C, D:

```bash
python main.py --csv data/data-2.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/B_rag_replay.yaml
python main.py --csv data/data-2.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/C_rag_replay.yaml
python main.py --csv data/data-2.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/D_rag_llm_grounded.yaml
```

For a simpler exploratory run, leave `replay_from_run: null` in D. Compare the
run reports under `experiments/runs/<run_id>/final_report.json`.

## Report fields to compare

- `best.cv_mean`, `best.cv_std`: best candidate cross-validation performance
- `best_candidate_origin`: `llm`, `evidence_compiler`, or `llm_evidence_design`
- `best_candidate_evidence_mapping`: exact supporting excerpts and matched operators
- `history[].evidence_mapping`: how each executed architecture maps to literature
- `research_influence.llm_grounded_candidates_executed`: D's successfully evaluated grounded proposals
- `research_influence.literature_derived_candidates_executed`: C's deterministic-template proposals
- `research_influence.generated_candidates_evaluated`: actual, not requested, evaluation count
- `evidence_translation_audit`: accepted and rejected proposals with validation reasons
- `run_wall_time_seconds_excluding_pdf`: end-to-end elapsed time including LLM work
- `research_evidence_sha256`: identical hashes mean frozen extracted evidence is identical

## Scientific limitations

- The validation is **not scientific fact checking** and does not show that a
  paper supports the entire architecture, only that a quoted methodological
  operation is also executed. The LLM may still select unhelpful compositions.
- Current WDBC dataset is a small tabular benchmark, not a clinically
  validated diagnosis system.
- Repeatedly comparing `holdout` scores from A/B/C/D undermines holdout
  independence. Do not use the holdout to choose the winning research method.
  For publishable claims, use fresh outer splits/nested evaluation or an
  independent final test set.
- Successful experiments require local Ollama, and live model generation and
  live literature retrieval have **not** been tested in this environment.
- D can have lower actual evaluation count if Qwen fails grounding checks;
  include actual candidate counts and compute budget in analyses.
