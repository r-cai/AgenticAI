"""Summarize multiple A/B/C final_report.json files to CSV without mixing holdouts.

Usage: python -m examples.compare_runs experiments/runs --out evidence_ablation_summary.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize(root):
    rows=[]
    for path in sorted(Path(root).glob('*/final_report.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        cfg_path=path.parent/'config_snapshot.json'
        cfg=json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        inf=r.get('research_influence') or {}
        evidence_comp=cfg.get('evidence_composition',{}).get('enabled',False)
        rag=cfg.get('rag',{}).get('enabled',False)
        evidence=cfg.get('evidence',{}).get('enabled',False)
        arm='C_rag_evidence_compiler' if rag and evidence and evidence_comp else ('B_rag_context_only' if rag else 'A_no_rag')
        history=r.get('history') or []
        rows.append({'run_id':r.get('run_id'), 'treatment':arm, 'seed':cfg.get('seed'),
                     'dataset':r.get('dataset'), 'cv_metric':r.get('scoring'),
                     'cv_best':(r.get('best') or {}).get('cv_mean'),
                     'cv_baseline':r.get('baseline_cv_score'),
                     'holdout_descriptive_only':(r.get('holdout') or {}).get('score'),
                     'evaluated_candidates':sum(h.get('iteration',0)>0 for h in history),
                     'derived_candidates':inf.get('literature_derived_candidates_executed',0),
                     'evidence_record_count':inf.get('evidence_records_extracted',0),
                     'winner_referenced_evidence':inf.get('best_is_literature_derived',False),
                     'winner_is_composition':inf.get('best_is_composition',False),
                     'run_wall_seconds':r.get('run_wall_time_seconds_excluding_pdf')})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('runs_dir')
    p.add_argument('--out',default='evidence_ablation_summary.csv')
    args=p.parse_args()
    table=summarize(args.runs_dir)
    if table.empty:
        raise SystemExit('No final_report.json files found')
    table.to_csv(args.out,index=False)
    print(table.to_string(index=False))
    print('Saved',args.out)


if __name__=='__main__':main()
