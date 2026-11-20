**Update — one paper per JSON:** The current storage model is detailed in [`PAPER_JSON_KNOWLEDGE_BASE.md`](PAPER_JSON_KNOWLEDGE_BASE.md). JSON documents under `<AgenticAI>/RAG_Knowledge_Base/papers/` are the portable records; SQLite is only a rebuildable index. The remainder of this document describes the original recovery and experimental replay protocol.

# Persistent RAG Literature Knowledge Base — Restoration & Use

## What was actually recovered

I inspected the pipeline ZIP archives and stand-alone reports available in the current workspace, including the older `AgenticResearchPipeline.zip`, the Level-2 A/B/C/D revisions, and `Ruoxin_Level2.zip`. The original ZIP contained 12 `rag/cache/*.json` batches of real API-retrieved metadata with **78 distinct paper abstracts**. Ten historical `final_report*.json` files contained **32 distinct extracted claim records**. Comparing claims to the available abstracts recovers **21 verbatim excerpt matches**. **11 claims cannot yet be matched to a recovered abstract**; these remain in the repository but are not used as valid evidence. A total of **85 distinct paper/citation records** are indexed: 78 with abstracts and 7 citation-only.

These are *historical records*, not new live literature searches. Matching a quote to its stored abstract verifies the quotation's provenance only; it does not verify the scientific correctness or applicability of the paper. Existing method, modality, and experimental eligibility checks remain in force. Source file names are retained as provenance. No paper or abstract was synthesized to fill gaps.

The user's Windows `experiments/runs/` history is **not available** within these archives. Run the importer below on the Windows machine to recover any additional `research.json`, `evidence.json`, `eligible_evidence.json`, `research_next.json`, and `rag/cache` files from earlier runs and package versions.

## Persistent location and automatic initialization

The package includes the restored seed SQLite index and 85 historical paper JSON records. On first use from any project directory **under `C:\Users\charl\anaconda3\AgenticAI`**, the pipeline creates a **shared** directory:

```text
C:\Users\charl\anaconda3\AgenticAI\RAG_Knowledge_Base\
  papers\                 # human-readable JSON; one paper/citation per file
  literature.sqlite3       # optional rebuildable search/deduplication index
```

The folder is automatically created without needing to add a `rag/cache` folder. Set `AGENTIC_RAG_KB_DIR` if the pipeline is launched outside the `AgenticAI` tree. Every new retrieved paper is merged into the same collection (duplicate identifiers are consolidated). Claims are eligible for D only when their excerpts can be found in the stored abstract and pass the existing evidence-to-method validation. Malformed source records are retained for review and not sent to the LLM. See [PAPER_JSON_KNOWLEDGE_BASE.md](PAPER_JSON_KNOWLEDGE_BASE.md) for schema, verification details, migration and repair.

`rag/cache/` is still the old exact-query cache for compatibility, not the persistent paper database.

## On your Windows machine: import every historical pipeline

Extract the new ZIP and open PowerShell in its directory containing `main.py`.

**1. Check the bundled records (no internet or Ollama):**

```powershell
python -m scripts.restore_literature --summary
```

**2. Scan the directory containing *all* previously extracted pipeline versions and ZIP archives:**

```powershell
python -m scripts.restore_literature --scan "C:\Users\charl\anaconda3\AgenticAI" --output literature_restore_audit.json
```

The importer reads recognized JSON files **inside ZIPs without extracting them**, and reads saved paper and evidence JSON files from experiment directories. It skips unrelated CSVs, code, medical records, and raw datasets. It does **not** execute code in older ZIPs. Source paths in the audit identify what contributed. If some earlier runs live outside `AgenticAI`, repeat `--scan` for those folders, or use `--source` for individual ZIP or run directory:

```powershell
python -m scripts.restore_literature --source "C:\OtherBackups\AgenticResearchPipeline.zip"
python -m scripts.restore_literature --source "C:\OldProject\experiments\runs"
python -m scripts.restore_literature --summary
```

Runs are not restored by the *report files alone*: the importer can recover citations and excerpt records from a final report but it needs `research.json` or an original query cache to recover the full supporting abstract. Unsupported historical claims remain quarantined until a matching abstract is imported.

## For a new research task: search old knowledge AND search new papers

The B/C/D YAML files include:

```yaml
rag:
  enabled: true
  knowledge_base:
    enabled: true
    refresh_mode: on_unseen_query
    min_local_score: 0.06
    cooldown_hours: 24
```

`on_unseen_query` (default): search the local database first; for a **new** planner query, attempt an online search to add fresh papers even when there are enough local matches. Recently attempted queries (including failed ones) have a 24-hour cooldown to limit 429 and timeout storms. When online retrieval fails, local papers still work. Other modes:

- `offline` — only existing repository and exact-query cache; no live APIs.
- `on_miss` — online only if too few relevant local results exist.
- `always` — attempt a new online search even for an already used query.

To override for one run without editing YAML:

```powershell
python main.py --csv data/data-cbis-ddsm.csv --target diagnosis --task "Classify benign and malignant breast lesions using mammographic metadata, feature selection, ensemble learning, and tabular machine learning" --config configs/D_rag_llm_grounded.yaml --kb-mode offline
```

This local-only mode avoids arXiv and Semantic Scholar errors, but still needs a working local Qwen generator. It is not guaranteed to have eligible evidence for every future task.

## Freeze one literature corpus for matched B/C/D comparisons

The ZIP includes two ready-made snapshots, created from *recovered real metadata* with no live retrieval:

- `examples/frozen_cbis_example/`: 25 abstracts, 12 excerpt-matched claims, 4 selected eligible method records.
- `examples/frozen_wdbc_example/`: 25 abstracts, 21 excerpt-matched claims, 4 selected eligible method records.

Use these to rerun D immediately, without trying online retrieval:

```powershell
python main.py --csv data/data-cbis-ddsm.csv --target diagnosis --task "Classify benign and malignant breast lesions using mammographic metadata, feature selection, ensemble learning, and tabular machine learning" --config configs/D_rag_llm_grounded.yaml --rag-replay examples/frozen_cbis_example
```

```powershell
python main.py --csv data/data-3.csv --target diagnosis --task "Predict malignant versus benign tumors using WDBC measurements" --config configs/D_rag_llm_grounded.yaml --rag-replay examples/frozen_wdbc_example
```

After importing your Windows-run history, you can create **new** task-specific frozen corpora that include any additional recovered papers. Use a new snapshot folder to avoid overwriting a referenced experiment corpus:

```powershell
python -m scripts.freeze_literature --query "Classify benign and malignant breast lesions using mammographic metadata, feature selection, ensemble learning, and tabular machine learning" --output experiments/frozen/cbis_v2 --top-k 25
```

Now run **B, C, and D against the same snapshot** by adding `--rag-replay experiments/frozen/cbis_v2` to each otherwise-identical command. In replay mode, the pipeline does not refresh the snapshot mid-experiment; its SHA-256 manifest and run artifacts help verify that the input literature is identical. Keep CV folds, seeds, candidate budgets and task description matched too. Treat the holdout as one shared held-out sample, not independent repeated testing.

## Guarantees, limits and interpretation

1. Every stored paper has source provenance. Duplicate imports do not increase unique paper counts.
2. Papers with no abstract are retained as citation metadata, **not provided** to the LLM as research context.
3. Evidence records with no verbatim abstract match are preserved for audit, **not provided** to Experiment D as valid citations.
4. Reused excerpt-matched records still face all existing structural method and modality checks. An abstract quote does not prove a technique is effective for CBIS-DDSM.
5. Live research expands the repository for future tasks, but makes the corpus time-dependent. Use an immutable `--rag-replay` snapshot for scientific comparisons.
6. The revised code has offline unit tests. Live Qwen operation and live academic API availability in the user's Windows environment have **not** been verified by this update.
7. If a future D run reports `insufficient_eligible_evidence`, the selected corpus is inadequate even if the knowledge base itself contains many papers; freeze a more targeted corpus or obtain supporting studies. Never fill the gap with fabricated method claims.

The ZIP also includes `data/data-3.csv`, reconstructed from the previously supplied `data-2.csv` by removing the `id` and empty `Unnamed: 32` columns (569 rows, 30 numeric predictors), alongside `data/data-cbis-ddsm.csv`. If you maintain a canonical local copy of `data-3.csv`, use that copy consistently for all four experiment arms.

The separate named A/B/C/D reporting feature remains enabled. The ZIP also preserves the earlier D Guided Design, empty-evidence recovery and CBIS-DDSM data handling fixes.
