# November 2026 — evidence-to-composition compiler

Based on user-uploaded `AgenticResearchPipeline.zip` (preserved independently).

## Implemented

- `rag/evidence_compiler.py`: source-excerpt method mapping, relevance checks,
  strict treatment of unsupported scientific methods, and candidate provenance.
- `orchestration/evidence_integration.py`: independent, budget-neutral execution
  path for automatic conversion of paper methods into Level-2 compositions;
  checks LLM-generated citations and architecture compatibility.
- `orchestration/controller.py`: integrates new candidates into the existing
  iterative generator/critic loop, with audit records, origin, full history,
  CV wall time, and experiment-wide identifier exclusions.
- `agents/schemas.py`: records candidate provenance and evaluation wall time.
- `orchestration/reporter.py`: shows evidence translation acceptance/rejections
  and research influence statistics in the PDF output.
- Four controlled-experiment YAML files under `configs/`.
- `requirements.txt`, offline demonstration, report comparison tool, and 12
  automated tests, including a mocked full-controller round.

## Scientific integrity

- Extracted excerpt must explicitly mention the method being implemented.
- Generic feature selection is not falsely represented as RFE.
- The RAG arm receives no additional CV evaluation budget.
- Paper-reported method is distinguished from the new compiler's hyperparameters
  and component choices; those are *hypotheses*, not paper reproductions.
- Keyword-only fallback evidence is not used to create literature-verified models.
- Image-specific methods are not directly transferred to numerical WDBC data.
- No arbitrary LLM-generated Python is executed.
- Cross-validation and model choice remain training-side only; holdout remains
  one-time after model selection.

## Not claimed

- No live Ollama or external scholarly retrieval was run in this environment.
- No demonstrated RAG performance improvement yet. Automated translation was
  tested; its efficacy requires controlled A/B/C experiments and new datasets.
- No guarantee that matching an abstract's stated method equals exact
  reproduction of a paper's algorithm.
