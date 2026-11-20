# CBIS-DDSM + reproducible A/B/C/D comparison (patch)

## Fixing `Configured excluded columns absent from dataset: ['id']`

`data/data-2.csv` has an identifier column, while `data/data-cbis-ddsm.csv` does not. This version drops only columns present in the input and logs absent optional exclusions in `data_exclusions.json`. It still rejects excluding the target. Configure `data.missing_exclusions: error` for strict enforcement instead.

### Command

From the folder containing `main.py`:

```powershell
python main.py --csv data/data-cbis-ddsm.csv --target diagnosis --task "Predict whether a patient has at least one malignant lesion from CBIS-DDSM mammography metadata" --config configs/A_no_rag.yaml
```

The same command works with the B, C, and D config paths. The output directory is printed as an **absolute path**, and appears under `experiments/runs/` relative to the current terminal working directory. The included `data/data-cbis-ddsm.csv` has one row per patient, 49 numeric variables, `diagnosis` B/M. It is NOT a mammogram image dataset, and it is NOT equivalent to the Wisconsin 30-feature cell-nucleus dataset. Its target is *patient has at least one malignant annotated lesion*.

## Candidate budget recovery

The configurations use `loop.candidates_per_iteration: 3`, `loop.max_batch_rounds: 3`, and patience 4 (over 3 iterations). After invalid/duplicate generator proposals, the controller tries to fill unused slots with fresh proposals. It never trains more than 3 candidates per iteration. The report records `configured_candidate_budget`, `attempted_candidate_slots`, `generated_candidates_evaluated`, `successful_generated_candidates`, `unfilled_candidate_slots` and `generation_calls` (outer batch generator calls, not inner Ollama retries). **Equal configured budgets do not guarantee equal actual evaluated counts**. Do not claim a fair A/B/C/D comparison until successful training counts are comparable. Extra model-generation retries count toward the computational cost.

## Evidence relevance and diversity

`rag/evidence_selection.py` deterministically filters source claims that do not support a registered structural operation, cautionary papers, and papers clearly about an incompatible image-based method. Records are grouped by supported method, round-robin-selected, and capped per method. The generator also receives previously represented composition kinds to encourage different valid architectures; a soft kind-cap can defer overrepresented kinds, but deferred valid candidates are used if needed to avoid wasting the model budget. Evidence selection is a heuristic for operational relevance, *not a scientific verification of papers*. It does not guarantee architecture novelty or performance gains.

All extracted evidence and selection decisions are preserved in `evidence.json`, `eligible_evidence.json`, `evidence_selection_audit.json`, and `final_report.json`. B/C/D should preferably replay the same frozen `research.json` and `evidence.json` using their `rag.replay_from_run` setting.

## Important experiment design warnings

1. `data-cbis-ddsm.csv` is aggregated at the patient level. It avoids repeated patient rows crossing a random split, but does **not** preserve the original TCIA split. Be explicit about this in a paper. For lesion-level analysis, use grouping with patient ID and the original metadata files instead.
2. Do not use the previously inspected 114-row Wisconsin holdout to select A/B/C/D or assess the new method's advantage. Use separate outer folds or a fresh independent holdout.
3. Always report real training counts and computation times, including failed generation/retries. For comparing models, keep the same dataset split, scoring, CV folds, model registry, and freeze literature.
4. If no compatible method is found in the literature, it is valid for a RAG arm to have zero eligible evidence; record this rather than forcing irrelevant citations.
