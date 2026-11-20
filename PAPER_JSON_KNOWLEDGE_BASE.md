# Shared RAG knowledge base — one readable JSON file per paper

## Location on your Windows computer

The default for experiments started from your pipeline folders is:

```text
C:\Users\charl\anaconda3\AgenticAI\RAG_Knowledge_Base\
    papers\
        a-structured-combination-of-ensemble-classifier...--p_....json
        a-new-computer-aided-diagnosis-system...--p_....json
        ... (one JSON file per distinct paper)
    literature.sqlite3      # rebuildable deduplication/search index, not the portable source
```

The pipeline **auto-detects the enclosing directory named `AgenticAI`** when it is launched from there. It uses its `RAG_Knowledge_Base` subdirectory across **all** extracted pipeline versions. To choose this location explicitly in PowerShell (e.g. when launching from a different directory), set:

```powershell
$env:AGENTIC_RAG_KB_DIR = "C:\Users\charl\anaconda3\AgenticAI\RAG_Knowledge_Base"
```

This override applies to the current PowerShell session. The directory can also be configured by setting `rag.knowledge_base.directory` in each B/C/D YAML file.

All papers, both restored and newly discovered, are automatically saved in **the same `papers` folder**, with one indented UTF-8 JSON document per paper. You do **not** have to manually create the folder. A paper found again does not create a second file; its existing document gains new provenance entries, and a longer abstract can replace an incomplete one. The filename contains a readable title plus a stable paper key.

## What's in each JSON file?

```json
{
  "schema_version": 1,
  "paper_key": "p_<stable hash>",
  "title": "Title exactly as reported by the source",
  "abstract": "Original retrieved abstract",
  "year": "2024",
  "doi": "",
  "url": "https://arxiv.org/abs/...",
  "source": "arxiv",
  "id": "https://arxiv.org/abs/...",
  "first_seen": "...",
  "last_seen": "...",
  "aliases": ["arxiv:...", "title:..."],
  "provenance": ["historical cache/ZIP source", "new retrieval query"],
  "evidence": [
    {
      "evidence_id": "ev_...",
      "record": {"source_excerpt": "Literal quote from abstract", "paper_title": "..."},
      "verified": true,
      "verification_reason": "excerpt_matches_stored_abstract",
      "provenance": ["experiment run directory"]
    }
  ],
  "verification": {
    "status": "metadata_checked",
    "checks": {
      "title_present": true,
      "abstract_present": true,
      "identifier_present": true,
      "source_identifier_consistent": true
    },
    "scope": "local metadata structure and excerpt matching only",
    "independently_verified_with_publisher": false,
    "excerpt_matched_count": 1,
    "unverified_evidence_count": 0
  }
}
```

Actual files contain real recovered metadata; the example above shows the schema only.

**Two-stage validation:** The metadata checker checks bibliographic completeness, identifier syntax, and whether the URL is consistent with the stated provider. Extracted scientific evidence separately requires the literal source excerpt to occur in the stored abstract, and the existing method/model/modality validator must also approve it. Unmatched claims are stored with `verified: false` and are not used by strict D. Malformed or source-inconsistent paper records remain readable with `verification.status: needs_review` and are excluded from local retrieval. Citation-only records are retained but not passed to the LLM.

This is **not an independent online verification against the publisher**, and it does not establish that a paper is scientifically correct or that the result transfers to your dataset. For higher assurance, a separate publisher/DOI-level audit should be performed when the APIs are available. Do not edit `verified` in JSON to authorize a claim: import/rebuild recomputes evidence support from the abstract.

## First run: recover all earlier pipeline versions

Extract the new pipeline ZIP into the `AgenticAI` folder and open PowerShell where `main.py` is located.

```powershell
python -m scripts.restore_literature --summary
python -m scripts.restore_literature --scan "C:\Users\charl\anaconda3\AgenticAI" --output literature_restore_audit.json
python -m scripts.restore_literature --summary
```

The first command automatically installs and exports the 85 historical paper/citation records recovered in the current workspace (78 with abstracts, 7 citation-only). The scan then searches all accessible older extracted pipelines and ZIPs for `rag/cache/*.json`, saved `research.json`, `evidence.json`, `final_report*.json`, *and paper-per-JSON folders*, and merges any additional papers and source evidence. It does not invent abstracts from citations alone. Check `literature_restore_audit.json` for all scanned sources and errors. Rerunning the scan is safe; results are deduplicated.

**Import your earlier SQLite knowledge base too**, if your previous ZIP's pipeline created one under your Windows home directory:

```powershell
python -m scripts.restore_literature --legacy-db "C:\Users\charl\.agentic_research_kb\literature.sqlite3" --output legacy_db_import_audit.json
```

Use this last command only if that old file exists; otherwise it prints a clear missing-file warning. It reads the old database without modifying it and preserves paper and evidence provenance.

## When you start a new task

Your existing A/B/C/D commands remain unchanged. For B, C and D, the retriever first searches the accumulated JSON-backed repository, then uses the online scholarly APIs for previously unseen queries (by default `on_unseen_query`). Every accepted new paper is **immediately written to the shared `papers` folder**, even before Qwen evidence extraction; when new excerpt-verified evidence appears, the same paper JSON is updated. API rate limits or failures do not delete existing papers. `--kb-mode offline` uses only stored papers, so D can still run if the stored evidence supports its selected method.

For fair experiments, freeze the current corpus and pass `--rag-replay` to B/C/D rather than comparing results against a repository that continues to grow between runs.

## Repair, inspection and portability

```powershell
python -m scripts.restore_literature --summary
python -m scripts.restore_literature --rebuild-index
```

`--rebuild-index` scans every paper JSON and rechecks its evidence excerpts; it does not simply trust the old `verified` flag. If `literature.sqlite3` is lost, the next normal pipeline run automatically reconstructs the index from the JSON files. The JSON paper directory can be copied to another machine and reopened; no proprietary database export is required.

Do not edit paper keys or delete old JSON during a running experiment. Keep backups/snapshots of the repository as experiments grow.
