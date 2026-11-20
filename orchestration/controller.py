import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from agents.deep import generate
from agents.qwen import critique
from agents.schemas import (
    CandidateConfig,
    CandidateResult,
    ModelConfig,
    PreprocessingConfig,
)
from data.profiler import load_and_profile, profile_dataset
from ml.evaluator import evaluate_holdout
from ml.trainer import run_cv
from rag.evidence import extract_evidence
from rag.evidence_selection import select_evidence
from rag.evidence_recovery import recover_evidence
from rag.retriever import _retrieve_queries, _kb_open
from orchestration.evidence_integration import generate_batch
from orchestration.report_naming import save_report_copies
from rag.retriever import build_followup_brief, build_research_brief

DEFAULTS = {
    "seed": 42,
    "test_size": 0.2,
    "data": {"exclude_columns": [], "missing_exclusions": "ignore"},
    "cv": {
        "folds": 5,
        "scoring": "f1_macro",
    },
    "loop": {
        "max_iterations": 3,
        "candidates_per_iteration": 3,
        "patience": 2,
        "min_delta": 0.002,
        "max_failures": 2,
        "max_batch_rounds": 1,
    },
    "agents": {
        "generator": {
            "model": "qwen3.5:9b",
            "temperature": 0.2,
        },
        "critic": {
            "enabled": True,
            "model": "qwen3:4b",
            "temperature": 0.2,
            "think": False,
        },
    },
    "rag": {
        "enabled": True,
        "top_k": 5,
        "timeout_sec": 20,
    },
    "evidence": {
        "enabled": True,
        "mode": "llm",
        "model": "qwen3:4b",
        "max_papers": 5,
    },
    "evidence_composition": {
        "enabled": True,
        "reserve_slots": 1,
        "allow_keyword": False,
    },
}

BASELINE = CandidateConfig(
    rationale="Default random forest baseline",
    preprocessing=PreprocessingConfig(),
    model=ModelConfig(name="random_forest"),
)


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_config(path: str | Path) -> dict:
    cfg = deepcopy(DEFAULTS)
    path = Path(path)

    if not path.is_file():
        print(f"[config] {path} not found; using defaults")
        return cfg

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("config.yaml must contain a YAML mapping")

    _merge(cfg, user_cfg)
    print(f"[config] Loaded {path.resolve()}")

    return cfg


def _save(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _evaluate(candidate, candidate_id, iteration, train_df, target, cfg, *, origin=None, evidence_mapping=None) -> CandidateResult:
    """Run CV for one candidate. A bad config becomes an error result, not a crash."""
    start = time.perf_counter()
    try:
        metrics = run_cv(candidate.model_dump(), train_df, target, cfg)
        metrics["cv_wall_time_seconds"] = round(time.perf_counter() - start, 4)
        return CandidateResult(
            candidate_id=candidate_id, iteration=iteration, config=candidate,
            origin=origin, evidence_mapping=evidence_mapping or {}, **metrics
        )
    except Exception as e:
        return CandidateResult(
            candidate_id=candidate_id,
            iteration=iteration,
            config=candidate, origin=origin, evidence_mapping=evidence_mapping or {},
            cv_wall_time_seconds=round(time.perf_counter() - start, 4),
            error=f"{type(e).__name__}: {e}",
        )


def _is_valid(r: CandidateResult) -> bool:
    return r.error is None and r.mean_cv_score is not None


def _history(results: list[CandidateResult]) -> list[dict]:
    """Compact view of everything tried so far, for the generator.
    Fine for a handful of iterations; truncate to top-N later if prompts grow."""
    return [
        {
            "id": r.candidate_id,
            "model": r.config.model.model_dump() if r.config.model else None,
            "composition": r.config.composition.model_dump() if r.config.composition else None,
            "evidence_ids": r.config.evidence_ids,
            "origin": r.origin,
            "preprocessing": r.config.preprocessing.model_dump(),
            "cv_mean": r.mean_cv_score,
            "train_cv_gap": r.train_cv_gap,
            "error": r.error,
            "convergence_warning_count": r.convergence_warning_count,
        }
        for r in results
    ]


def _collect_research(future, timeout_sec: int):
    """Wait for a background RAG job, but never let it block or crash the run."""
    if future is None:
        return []
    try:
        return future.result(timeout=timeout_sec)
    except Exception as e:  # includes timeout
        print(f"[rag] skipped: {type(e).__name__}: {e}")
        return []


def _followup_research(feedback: dict, description: str, profile: dict, cfg: dict,
                       seen_titles: set, history=None, previous_queries=None, plan_path=None):
    """Retrieval targeted at the critic's latest feedback, bounded by the RAG timeout."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            build_followup_brief, feedback, description, profile,
            k=cfg["rag"]["top_k"], exclude=seen_titles,
            history=history, previous_queries=previous_queries,
            planner_model=cfg["rag"].get("planner_model", "qwen3:4b"),
            plan_path=plan_path,
            knowledge_base=cfg["rag"].get("knowledge_base", {}),
        )
        return _collect_research(future, cfg["rag"]["timeout_sec"])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _extract(papers: list[dict], cfg: dict) -> list[dict]:
    """Reuse verified prior excerpts; extract additional claims from new papers.

    Historical evidence is never used merely because a report set
    'source_grounded=True'. The knowledge base requires verbatim excerpt
    support in a stored abstract before returning a claim here.
    """
    if not cfg["rag"]["enabled"] or not cfg["evidence"]["enabled"]:
        return []
    kb = _kb_open(cfg['rag'].get('knowledge_base', {}))
    existing = kb.verified_evidence_for(papers) if kb else []
    evidenced_titles = {x['paper_title'].casefold() for x in existing}
    to_extract = [p for p in papers if str(p.get('title', '')).casefold() not in evidenced_titles]
    if existing:
        print(f"[rag/kb] reused {len(existing)} excerpt-verified evidence records")
    fresh = []
    if to_extract:
        try:
            fresh = extract_evidence(
                to_extract,
                model=cfg["evidence"]["model"],
                mode=cfg["evidence"]["mode"],
                max_papers=cfg["evidence"]["max_papers"],
            )
        except Exception as exc:
            print(f"[evidence] extraction skipped: {type(exc).__name__}: {exc}")
    if kb:
        kb.add_papers(papers, origin='current_run:research_context')
        if fresh:
            kb.add_evidence(fresh, origin='current_run:extraction')
    return list({item['evidence_id']: item for item in (existing + fresh)}.values())


def run_experiment(csv_path: str,
                   target: str,
                   description: str,
                   config_path: str = "config.yaml", *,
                   replay_from_run: str | None = None,
                   kb_mode: str | None = None) -> dict:
    run_start = time.perf_counter()
    cfg = _load_config(config_path)
    if replay_from_run is not None:
        cfg['rag']['replay_from_run'] = replay_from_run
        print(f"[rag] CLI frozen replay: {replay_from_run}")
    if kb_mode is not None:
        if kb_mode not in ('offline', 'on_miss', 'on_unseen_query', 'always'):
            raise ValueError('Invalid kb_mode')
        cfg['rag'].setdefault('knowledge_base', {})['refresh_mode'] = kb_mode
        print(f"[rag/kb] CLI refresh mode: {kb_mode}")
    loop = cfg["loop"]
    control_cfg = cfg.get("controlled_study", {})
    target_budget = loop.get("target_evaluated_candidates")
    if target_budget is not None:
        target_budget = int(target_budget)
        if target_budget < 1:
            raise ValueError("loop.target_evaluated_candidates must be positive")
        if int(loop["max_iterations"]) * int(loop["candidates_per_iteration"]) < target_budget:
            raise ValueError("Maximum generation slots are fewer than target_evaluated_candidates")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("experiments/runs") / run_id
    run_dir.mkdir(parents=True)
    _save(run_dir / "config_snapshot.json", cfg)
    print(f"Run {run_id}")
    print(f"[output] reports will be saved to {run_dir.resolve()}")

    # 1. Load, split (holdout set aside), profile training data only.
    # Row positions are preserved by train_test_split. Hash of raw CSV bytes
    # additionally guards against silent edits between experiment arms.
    dataset_sha256 = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    train_df, test_df, profile = load_and_profile(
        csv_path, target, test_size=cfg["test_size"], seed=cfg["seed"],
        cv_folds=cfg["cv"]["folds"],
    )
    # Apply identical, experiment-level column exclusions to ALL search arms.
    # This avoids giving the literature arm an unfair advantage by allowing
    # only its candidates to remove an identifier such as WDBC's `id`.
    data_cfg = cfg.get("data", {})
    global_exclusions = data_cfg.get("exclude_columns", [])
    if not isinstance(global_exclusions, list) or any(not isinstance(c, str) for c in global_exclusions):
        raise ValueError("data.exclude_columns must be a list of column names")
    if target in global_exclusions:
        raise ValueError("Cannot exclude the prediction target")
    missing_policy = data_cfg.get("missing_exclusions", "ignore")
    if missing_policy not in ("ignore", "error"):
        raise ValueError("data.missing_exclusions must be ignore or error")
    missing = sorted(set(global_exclusions) - set(train_df.columns))
    if missing and missing_policy == "error":
        raise ValueError(f"Configured excluded columns absent from dataset: {missing}")
    exclusions_present = [c for c in global_exclusions if c in train_df.columns]
    if missing:
        print(f"[data] optional excluded columns not present; skipped: {missing}")
    if exclusions_present:
        train_df = train_df.drop(columns=exclusions_present)
        test_df = test_df.drop(columns=exclusions_present)
        profile = profile_dataset(train_df, target)
    exclusion_audit = {"requested": global_exclusions, "removed": exclusions_present,
                       "absent_optional": missing, "missing_policy": missing_policy}
    split_manifest = {
        "seed": int(cfg["seed"]), "test_size": float(cfg["test_size"]),
        "dataset_sha256": dataset_sha256,
        "train_row_indices": [int(x) for x in train_df.index],
        "test_row_indices": [int(x) for x in test_df.index],
        "target": target,
    }
    split_manifest["split_sha256"] = hashlib.sha256(json.dumps({
        "train": sorted(split_manifest["train_row_indices"]),
        "test": sorted(split_manifest["test_row_indices"]),
        "dataset_sha256": dataset_sha256,
    }, sort_keys=True).encode()).hexdigest()
    _save(run_dir / "split_manifest.json", split_manifest)
    _save(run_dir / "profile.json", profile)
    _save(run_dir / "data_exclusions.json", exclusion_audit)

    # 2. Literature can be replayed for paired B/C/D comparisons.
    # Replayed evidence is fixed across runs and NEVER refreshed mid-search.
    replay_dir = cfg.get("rag", {}).get("replay_from_run")
    if replay_dir and not cfg.get("rag", {}).get("enabled", False):
        raise ValueError("rag.replay_from_run requires rag.enabled=true")
    pool = ThreadPoolExecutor(max_workers=1)
    research_future = (
        pool.submit(
            build_research_brief,
            description,
            profile,
            k=cfg["rag"]["top_k"],
            planner_model=cfg["rag"].get("planner_model", "qwen3:4b"),
            plan_path=run_dir / "research_plan.json",
            knowledge_base=cfg["rag"].get("knowledge_base", {}),
        )
        if cfg["rag"]["enabled"] and not replay_dir else None
    )

    all_results: list[CandidateResult] = []
    baseline = _evaluate(BASELINE, "baseline", 0, train_df, target, cfg)
    all_results.append(baseline)
    print(f"Baseline ({BASELINE.model.name}): CV {_fmt(baseline.mean_cv_score)}"
          + (f"  ERROR {baseline.error}" if baseline.error else ""))

    if replay_dir:
        frozen = Path(replay_dir)
        try:
            research = json.loads((frozen / "research.json").read_text(encoding="utf-8"))
            evidence = json.loads((frozen / "evidence.json").read_text(encoding="utf-8"))
            if not isinstance(research, list) or not isinstance(evidence, list):
                raise ValueError("replay files must contain JSON arrays")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot replay literature from {frozen}: {exc}") from exc
        print(f"[rag] reusing frozen research/evidence corpus from {frozen}")
    else:
        research = _collect_research(research_future, cfg["rag"]["timeout_sec"])
        evidence = _extract(research, cfg)
    pool.shutdown(wait=False, cancel_futures=True)
    _save(run_dir / "research.json", research)
    all_evidence_catalog = {item["evidence_id"]: item for item in evidence}
    selector_cfg = cfg.get("evidence_selection", {})
    def pick_evidence():
        if not selector_cfg.get("enabled", False):
            return list(all_evidence_catalog.values()), []
        selection = select_evidence(list(all_evidence_catalog.values()),
            max_records=int(selector_cfg.get("max_records", 9)),
            max_per_method=int(selector_cfg.get("max_per_method", 2)),
            require_structural=bool(selector_cfg.get("require_structural", True)))
        return selection.records, selection.audit
    selected_evidence, selection_audit = pick_evidence()
    evidence_catalog = {item["evidence_id"]: item for item in selected_evidence}
    translation_audit = []
    _save(run_dir / "evidence.json", evidence)
    _save(run_dir / "eligible_evidence.json", selected_evidence)
    _save(run_dir / "evidence_selection_audit.json", selection_audit)
    if cfg['rag']['enabled'] and not replay_dir:
        kb = _kb_open(cfg['rag'].get('knowledge_base', {}))
        if kb:
            provenance = f'run:{run_id}:dataset:{Path(csv_path).name}'
            kb.add_papers(research, origin=provenance)
            kb.add_evidence(evidence, origin=provenance)
    corpus_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()
    print(f"Research context: {len(research)} item(s)")
    print(f"Grounded method records: {len(evidence)}; eligible/diverse: {len(evidence_catalog)}")

    # Strict D requires verifiable source-supported architectural evidence.
    # Do a BOUNDED targeted retrieval before generation when initial papers fail.
    # Never silently turn D into an ordinary no-RAG or context-only B run.
    strict_d = bool(cfg.get("llm_evidence_design", {}).get("enabled", False))
    recovery_audit = []
    if strict_d and not evidence_catalog and cfg["rag"]["enabled"] and not replay_dir:
        print("[rag] D: no eligible methods; searching targeted tabular-method literature")
        rescue = recover_evidence(
            description, profile, cfg, research, list(all_evidence_catalog.values()),
            retrieve_fn=(
                (lambda queries, description, k, exclude=None: _retrieve_queries(
                    queries, description, k, exclude=exclude,
                    knowledge_base=cfg['rag']['knowledge_base']))
                if 'knowledge_base' in cfg['rag'] else _retrieve_queries
            ),
            extract_fn=lambda papers, max_papers: extract_evidence(
                papers, model=cfg["evidence"]["model"],
                mode=cfg["evidence"]["mode"], max_papers=max_papers),
        )
        recovery_audit = rescue.attempts
        research.extend(rescue.papers)
        all_evidence_catalog.update({item["evidence_id"]: item for item in rescue.evidence})
        selected_evidence, selection_audit = pick_evidence()
        evidence_catalog = {item["evidence_id"]: item for item in selected_evidence}
        evidence = list(all_evidence_catalog.values())
        kb = _kb_open(cfg['rag'].get('knowledge_base', {}))
        if kb:
            kb.add_papers(rescue.papers, origin=f'run:{run_id}:dataset:{Path(csv_path).name}:D_recovery')
            kb.add_evidence(rescue.evidence, origin=f'run:{run_id}:dataset:{Path(csv_path).name}:D_recovery')
        _save(run_dir / "research.json", research)
        _save(run_dir / "evidence.json", evidence)
        _save(run_dir / "eligible_evidence.json", selected_evidence)
        _save(run_dir / "evidence_selection_audit.json", selection_audit)
        print(f"[rag] D rescue: +{len(rescue.papers)} papers, +{len(rescue.evidence)} claims, "
              f"{len(evidence_catalog)} eligible")
    elif strict_d and not evidence_catalog and replay_dir:
        print("[rag] D replay has no eligible evidence; frozen corpus will NOT be changed")
    _save(run_dir / "evidence_recovery_audit.json", recovery_audit)
    corpus_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()

    # 3. Loop: generate -> train -> critique -> RAG on the feedback -> generate ...
    best_score = baseline.mean_cv_score if _is_valid(baseline) else float("-inf")
    seen_titles = {p["title"] for p in research}

    previous_queries = []
    initial_plan = run_dir / "research_plan.json"
    if initial_plan.exists():
        previous_queries.extend(json.loads(initial_plan.read_text(encoding="utf-8")).get("queries", []))

    rag_rounds = (1 if research else 0) + sum(
        1 for attempt in recovery_audit if attempt.get("new_papers", 0) > 0
    )
    tried = {BASELINE.fingerprint()}
    feedback = None
    stale = failures = iterations_run = 0
    stop_reason = "max_iterations"
    iterations_limit = int(loop["max_iterations"])
    if strict_d and not evidence_catalog:
        stop_reason = "insufficient_eligible_evidence"
        iterations_limit = 0
        print("[experiment] Strict D skipped: no verified supporting operations. "
              "Baseline only; do NOT count this as a completed D experiment.")

    try:
        for i in range(1, iterations_limit + 1):
            iterations_run = i
            it_dir = run_dir / f"iter_{i:03d}"
            it_dir.mkdir()
            print(f"\n=== Iteration {i} ===")

            eligible_titles = {item.get("paper_title") for item in evidence_catalog.values()}
            filtered_research = ([p for p in research if p.get("title") in eligible_titles]
                                 if selector_cfg.get("enabled", False) else research)
            batch = generate_batch(
                evidence=list(evidence_catalog.values()),
                evidence_catalog=evidence_catalog,
                profile=profile, train_df=train_df, target=target,
                cfg=cfg, tried=tried, generate_fn=generate,
                generator_kwargs={
                    "profile": profile, "research": filtered_research,
                    "research_evidence": list(evidence_catalog.values()),
                    "description": description,
                    "feedback": feedback.model_dump() if feedback else None,
                    "history": _history(all_results),
                    "model": cfg["agents"]["generator"]["model"],
                    "temperature": cfg["agents"]["generator"]["temperature"],
                },
            )
            _save(it_dir / "proposal.json", batch.dump())
            _save(it_dir / "evidence_translation_audit.json", batch.audit)
            translation_audit.extend({"iteration": i, **entry} for entry in batch.audit)
            if batch.generator_error:
                print(f"Generator issue: {batch.generator_error}")
            if not batch.candidates:
                failures += 1
                print(f"No valid proposals ({failures}/{loop['max_failures']})")
                if failures >= loop["max_failures"]:
                    stop_reason = "generator_failed"
                    break
                continue
            failures = 0

            results = []
            for j, cand in enumerate(batch.candidates, 1):
                fp = cand.fingerprint()
                if fp in tried:
                    print(f"  c{j}: duplicate of an earlier config, skipped")
                    continue
                tried.add(fp)
                res = _evaluate(cand, f"iter{i:02d}_c{j}", i, train_df, target, cfg,
                                origin=batch.origins.get(fp), evidence_mapping=batch.mapping.get(fp))
                results.append(res)
                if res.error:
                    print(f"  {res.candidate_id} {cand.display_name}: ERROR {res.error}")
                else:
                    print(f"  {res.candidate_id} {cand.display_name}: CV {_fmt(res.mean_cv_score)} "
                          f"(+/- {_fmt(res.std_cv_score)}, gap {_fmt(res.train_cv_gap)})")
                    if res.convergence_warning_count:
                        print(f"    [warning] {res.convergence_warning_count} convergence warning(s)")
                # Equal *trained* candidate budget, not merely an equal request.
                # Never evaluate a seventh candidate when the registered budget is six.
                n_success_before = sum(r.iteration > 0 and _is_valid(r) for r in all_results)
                n_success_new = sum(_is_valid(r) for r in results)
                if target_budget is not None and n_success_before + n_success_new >= target_budget:
                    break

            prior_history = _history(all_results)  # everything before this iteration
            all_results.extend(results)
            _save(it_dir / "results.json", [r.model_dump() for r in results])

            if target_budget is not None and sum(
                r.iteration > 0 and _is_valid(r) for r in all_results
            ) >= target_budget:
                stop_reason = "target_evaluated_candidates_reached"
                print(f"[controlled] completed {target_budget} successfully evaluated candidates")
                break

            # Progress = beating the best score so far by more than min_delta
            iter_best = max(
                (r.mean_cv_score for r in results if _is_valid(r)),
                default=float("-inf"),
            )
            improved = iter_best > best_score + loop["min_delta"]
            best_score = max(best_score, iter_best)
            stale = 0 if improved else stale + 1

            new_feedback = False
            if results and cfg["agents"]["critic"].get("enabled", True):
                try:
                    feedback = critique(
                        proposals=batch.dump(),
                        results=[r.model_dump() for r in results],
                        history=prior_history,
                        research=research,
                        research_evidence=evidence,
                        model=cfg["agents"]["critic"]["model"],
                        temperature=cfg["agents"]["critic"]["temperature"],
                        think=cfg["agents"]["critic"]["think"],
                    )
                    new_feedback = True
                    _save(it_dir / "feedback.json", feedback.model_dump())
                except Exception as e:
                    feedback = None
                    print(f"Critic failed ({type(e).__name__}: {e}); continuing without feedback")

            # Stopping conditions (the controller decides; the critic is advisory)
            if stale >= loop["patience"]:
                stop_reason = "patience"
                break
            # if feedback and feedback.likely_converged and stale > 0:
            #     stop_reason = "critic_converged"
            #     break

            # Follow-up research uses critic feedback when available, otherwise
            # CV history and task context. RAG and critic can be toggled independently.
            if cfg["rag"]["enabled"] and not replay_dir and i < loop["max_iterations"]:
                fresh = _followup_research(
                    feedback.model_dump() if new_feedback else {}, description, profile, cfg, seen_titles,
                    history=_history(all_results), previous_queries=previous_queries,
                    plan_path=it_dir / "research_plan_next.json",
                )
                plan_file = it_dir / "research_plan_next.json"
                if plan_file.exists():
                    previous_queries.extend(json.loads(plan_file.read_text(encoding="utf-8")).get("queries", []))
                if fresh:
                    research = fresh
                    evidence = _extract(research, cfg)
                    all_evidence_catalog.update({item["evidence_id"]: item for item in evidence})
                    selected_evidence, selection_audit = pick_evidence()
                    evidence_catalog = {item["evidence_id"]: item for item in selected_evidence}
                    _save(run_dir / "eligible_evidence.json", selected_evidence)
                    _save(run_dir / "evidence_selection_audit.json", selection_audit)
                    seen_titles |= {p["title"] for p in fresh}
                    rag_rounds += 1
                    _save(it_dir / "research_next.json", fresh)
                    _save(it_dir / "evidence_next.json", evidence)
                    print(f"[rag] {len(fresh)} paper(s) relevant to this feedback go to iteration {i + 1}")
                else:
                    print("[rag] nothing new for this feedback; keeping previous research")
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nInterrupted: finishing with the results so far")

    evaluated_count = sum(r.iteration > 0 and _is_valid(r) for r in all_results)
    budget_complete = target_budget is None or evaluated_count == target_budget
    if target_budget is not None and not budget_complete:
        print(f"[controlled] INCOMPLETE: evaluated {evaluated_count}/{target_budget} candidates; "
              "exclude this run from matched-budget inference")
        if stop_reason == "max_iterations":
            stop_reason = "candidate_budget_unfilled"

    # 4. Final selection: best mean CV score (baseline included)
    valid = [r for r in all_results if _is_valid(r)]
    if not valid:
        raise RuntimeError("No candidate trained successfully; nothing to select.")
    best = max(valid, key=lambda r: r.mean_cv_score)
    print(f"\nSelected {best.candidate_id} ({best.config.display_name}), CV {_fmt(best.mean_cv_score)}")

    # 5. Single holdout evaluation: refit on all training data, test once
    holdout_predictions_path = run_dir / "holdout_predictions.csv"
    holdout = evaluate_holdout(best.config, train_df, test_df, target, cfg,
                               prediction_path=holdout_predictions_path,
                               prediction_metadata={
                                   "dataset_sha256": dataset_sha256,
                                   "split_sha256": split_manifest["split_sha256"],
                                   "seed": cfg["seed"],
                                   "candidate_id": best.candidate_id,
                               })
    if control_cfg.get("save_baseline_predictions", False):
        # Separate diagnostic only; it is not a model-selection criterion.
        evaluate_holdout(BASELINE, train_df, test_df, target, cfg,
                         prediction_path=run_dir / "baseline_holdout_predictions.csv",
                         prediction_metadata={
                             "dataset_sha256": dataset_sha256,
                             "split_sha256": split_manifest["split_sha256"],
                             "seed": cfg["seed"], "candidate_id": "baseline",
                         })

    # 6. Report
    report = {
        "run_id": run_id,
        "output_dir": str(run_dir.resolve()),
        "dataset": csv_path,
        "target": target,
        "description": description,
        "scoring": cfg["cv"]["scoring"],
        "baseline_cv_score": baseline.mean_cv_score,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_manifest["split_sha256"],
        "predictions_file": str(holdout_predictions_path),
        "controlled_study": {
            "enabled": target_budget is not None,
            "target_evaluated_candidates": target_budget,
            "successfully_evaluated_candidates": evaluated_count,
            "budget_complete": budget_complete,
            "suite_id": control_cfg.get("suite_id"),
            "frozen_corpus_sha256": control_cfg.get("frozen_corpus_sha256"),
        },
        "run_wall_time_seconds_excluding_pdf": round(time.perf_counter() - run_start, 4),
        "best": {
            "candidate_id": best.candidate_id,
            "iteration": best.iteration,
            "config": best.config.model_dump(),
            "cv_mean": best.mean_cv_score,
            "cv_std": best.std_cv_score,
        },
        "holdout": holdout,
        "cv_holdout_gap": round(best.mean_cv_score - holdout["score"], 4),
        "iterations_run": iterations_run,
        "stop_reason": stop_reason,
        "research_used": rag_rounds > 0,
        "critic_enabled": cfg["agents"]["critic"].get("enabled", True),
        "rag_enabled": cfg["rag"]["enabled"],
        "research_rounds": rag_rounds,
        "research_source": ("frozen_replay" if replay_dir else
                            "persistent_kb_offline" if cfg.get('rag', {}).get('knowledge_base', {}).get('refresh_mode') == 'offline' else
                            "persistent_kb_and_optional_online_retrieval"),
        "knowledge_base": ({"enabled": bool(cfg.get('rag', {}).get('knowledge_base', {}).get('enabled', True)),
                           "directory": str(_kb_open(cfg['rag'].get('knowledge_base', {})).documents_dir),
                           "index_database": str(_kb_open(cfg['rag'].get('knowledge_base', {})).path),
                           "storage_format": "one_json_per_paper",
                           "refresh_mode": cfg.get('rag', {}).get('knowledge_base', {}).get('refresh_mode', 'on_unseen_query')}
                          if not replay_dir and _kb_open(cfg['rag'].get('knowledge_base', {})) else
                          {"enabled": False, "note": "frozen replay does not update the shared knowledge base"}),
        "research_evidence_sha256": corpus_digest,
        "evidence_recovery_audit": recovery_audit,
        "strict_D_completed": (None if not strict_d else bool(evidence_catalog) and
                               any(r.origin == "llm_evidence_design" and _is_valid(r)
                                   for r in all_results)),
        "llm_evidence_design_enabled": bool(cfg.get("llm_evidence_design", {}).get("enabled", False)),
        "llm_grounded_strategy": cfg.get("llm_evidence_design", {}).get("strategy", "freeform"),
        "evidence_records": list(all_evidence_catalog.values()),
        "eligible_evidence_ids": sorted(evidence_catalog),
        "evidence_selection_audit": selection_audit,
        "data_exclusion_audit": exclusion_audit,
        "evidence_ids_selected": best.config.evidence_ids,
        "best_candidate_origin": best.origin,
        "best_candidate_evidence_mapping": best.evidence_mapping,
        "evidence_translation_audit": translation_audit,
        "evidence_composition_enabled": bool(cfg.get("evidence_composition", {}).get("enabled", False)
                                            and cfg["rag"]["enabled"] and cfg["evidence"]["enabled"]),
        "research_influence": {
            "evidence_records_extracted": len(all_evidence_catalog),
            "evidence_records_eligible": len(evidence_catalog),
            "accepted_translation_attempts": sum(a.get("status") == "accepted" for a in translation_audit),
            "rejected_translation_attempts": sum(a.get("status") == "rejected" for a in translation_audit),
            "accepted_compiler_translations": sum(a.get("status") == "accepted" and bool(a.get("evidence_id")) for a in translation_audit),
            "rejected_llm_proposals": sum(a.get("status") == "rejected" and a.get("origin") in ("llm", "llm_evidence_design") for a in translation_audit),
            "generation_calls": sum(a.get("llm_calls", 0) for a in translation_audit if a.get("origin") == "batch_budget"),
            "literature_derived_candidates_executed": sum(r.origin == "evidence_compiler" and _is_valid(r) for r in all_results),
            "literature_cited_candidates_executed": sum(bool(r.config.evidence_ids) and _is_valid(r) for r in all_results),
            "generated_candidates_evaluated": sum(r.iteration > 0 for r in all_results),
            "successful_generated_candidates": sum(r.iteration > 0 and _is_valid(r) for r in all_results),
            "configured_candidate_budget": int(target_budget or loop["max_iterations"] * loop["candidates_per_iteration"]),
            "attempted_candidate_slots": int(iterations_run * loop["candidates_per_iteration"]),
            "unfilled_candidate_slots": max(0, int((target_budget if target_budget is not None else iterations_run * loop["candidates_per_iteration"])) - evaluated_count),
            "not_run_due_to_early_stop": (0 if target_budget is not None else max(0, int((loop["max_iterations"] - iterations_run) * loop["candidates_per_iteration"]))),
            "best_is_literature_derived": bool(best.config.evidence_ids),
            "best_is_composition": best.config.composition is not None,
            "best_is_evidence_compiled": best.origin == "evidence_compiler",
            "best_is_llm_evidence_grounded": best.origin == "llm_evidence_design",
            "llm_grounded_candidates_executed": sum(
                r.origin == "llm_evidence_design" and _is_valid(r) for r in all_results
            ),
        },
        "history": [
            {
                "candidate_id": r.candidate_id,
                "iteration": r.iteration,
                "model": r.config.display_name,
                "evidence_ids": r.config.evidence_ids,
                "origin": r.origin,
                "evidence_mapping": r.evidence_mapping,
                "composition": r.config.composition.model_dump() if r.config.composition else None,
                "cv_mean": r.mean_cv_score,
                "cv_std": r.std_cv_score,
                "train_cv_gap": r.train_cv_gap,
                "cv_wall_time_seconds": r.cv_wall_time_seconds,
                "error": r.error,
            }
            for r in all_results
        ],
    }
    _save(run_dir / "final_report.json", report)

    # 7. PDF report built from the saved artifacts. reportlab is optional: a missing
    # package or a rendering problem must never fail a finished run.
    try:
        from orchestration.reporter import generate_pdf
        report["pdf_report"] = str(generate_pdf(run_dir))
        print(f"PDF report: {report['pdf_report']}")
    except Exception as e:
        print(f"[report] PDF skipped: {type(e).__name__}: {e}")

    # Preserve the original file, provide a convenient dataset/experiment alias,
    # and keep a unique run-ID archive so that reruns cannot erase history.
    paths = save_report_copies(report, csv_path=csv_path,
                               config_path=config_path, run_dir=run_dir)
    print(f"[report] Named latest: {Path(paths['latest_copy']).resolve()}")
    print(f"[report] Run archive:  {Path(paths['archived_copy']).resolve()}")
    return report


def print_summary(report: dict) -> None:
    b = report["best"]
    print("\n" + "=" * 60)
    model = b["config"].get("model")
    name = model["name"] if model else f"composition:{b['config']['composition']['kind']}"
    print(f"Best: {b['candidate_id']} ({name})")
    print(f"  CV {report['scoring']}: {_fmt(b['cv_mean'])} +/- {_fmt(b['cv_std'])}")
    print(f"  Baseline CV:    {_fmt(report['baseline_cv_score'])}")
    print(f"  Holdout:        {_fmt(report['holdout']['score'])} "
          f"(CV-holdout gap {report['cv_holdout_gap']:+.4f})")
    print(f"Stopped: {report['stop_reason']} after {report['iterations_run']} iteration(s)")
    if report.get("llm_evidence_design_enabled") and not report.get("strict_D_completed", True):
        print("[WARNING] Experiment D was INCOMPLETE: no valid grounded candidates. "
              "Any reported baseline score must NOT be interpreted as a D result.")
    print(f"Full report: experiments/runs/{report['run_id']}/final_report.json")
    if report.get("report_paths"):
        print(f"Named report: {report['report_paths']['latest_copy']}")
        print(f"Archived copy: {report['report_paths']['archived_copy']}")
    if report.get("pdf_report"):
        print(f"PDF report:  {report['pdf_report']}")
