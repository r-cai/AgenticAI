"""Create a reference-style Agentic ML experiment PDF from saved run JSON.

Usage: generate_pdf(run_dir) -> pathlib.Path. Requires reportlab.
"""
from __future__ import annotations
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether

NAVY = colors.HexColor('#244E69')
HEADER = colors.HexColor('#E8F0F6')
STRIPE = colors.HexColor('#F5F7FA')


def _load(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def _fmt(v):
    if v is None: return '—'
    if isinstance(v, float): return f'{v:.4f}'
    if isinstance(v, (list, tuple)): return ', '.join(map(str, v)) or 'none'
    if isinstance(v, bool): return str(v)
    if isinstance(v, dict): return ', '.join(f'{k}={_fmt(x)}' for k,x in v.items()) or 'default'
    return str(v)


def generate_pdf(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    report = _load(run_dir/'final_report.json')
    if not isinstance(report, dict) or not isinstance(report.get('best'), dict):
        raise ValueError('Missing or invalid final_report.json')
    config = _load(run_dir/'config_snapshot.json', {}) or {}
    profile = _load(run_dir/'profile.json', {}) or {}
    output = Path(output_path) if output_path else run_dir/'level2_experiment_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle2', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=colors.HexColor('#192D40'), spaceAfter=16))
    styles.add(ParagraphStyle(name='Section2', parent=styles['Heading2'], fontSize=12.4, leading=16, textColor=NAVY, spaceBefore=16, spaceAfter=8))
    styles.add(ParagraphStyle(name='Sub2', parent=styles['Heading3'], fontSize=10, leading=13, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'], fontSize=8.6, leading=12.4, spaceAfter=6, splitLongWords=True, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='Cell2', parent=styles['Body2'], fontSize=7.2, leading=10, spaceAfter=0))
    styles.add(ParagraphStyle(name='Small2', parent=styles['Body2'], fontSize=7.6, leading=10.5, spaceAfter=4))
    story=[]
    W=7.1*inch
    def P(t, style='Body2'):
        return Paragraph(escape(str(t)).replace('\n','<br/>'), styles[style])
    def add(t, style='Body2'): story.append(P(t, style))
    def detail(k,v):
        story.append(Paragraph(f'<b>{escape(str(k))}:</b> {escape(_fmt(v))}', styles['Body2']))
    def section(t): add(t,'Section2')
    def sub(t): add(t,'Sub2')
    def table(rows, widths=None):
        if not rows:return
        widths = widths or [W/len(rows[0])]*len(rows[0])
        cells=[[P(x,'Cell2') for x in row] for row in rows]
        t=Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),HEADER),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,STRIPE]),
            ('LINEBELOW',(0,0),(-1,0),.45,colors.HexColor('#BCCAD3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)
        ]))
        story.append(t)
    def papers(items, label):
        sub(label)
        if isinstance(items, dict): items=items.get('papers',[])
        if not items: add('No papers saved for this retrieval.'); return
        for n, paper in enumerate(items,1):
            if not isinstance(paper,dict): continue
            heading=f"{n}. {paper.get('title','Untitled')} ({paper.get('year') or 'n.d.'})  {paper.get('url') or ''}"
            story.append(KeepTogether([P(heading,'Body2'),P(paper.get('abstract') or 'Abstract unavailable.','Small2')]))
    def prep_text(candidate):
        prep=candidate.get('preprocessing') or {}
        return ('imputer='+str(prep.get('numeric_imputer','median'))+', scaler='+str(prep.get('numeric_scaler','none'))+
                ', encoder='+str(prep.get('categorical_encoder','onehot'))+
                ', dropped columns: '+_fmt(prep.get('drop_columns',[]))+
                (', categorical columns: '+_fmt(prep['categorical_columns']) if prep.get('categorical_columns') else ''))
    def model_text(candidate):
        composition = candidate.get('composition')
        if composition:
            return 'Composition: ' + json.dumps(composition, ensure_ascii=False, default=str) + '\\n' + prep_text(candidate)
        model = candidate.get('model') or {}
        return str(model.get('name','unknown')) + '\\n' + _fmt(model.get('hyperparameters',{})) + '\\n' + prep_text(candidate)

    def architecture(candidate):
        if candidate.get('composition'):
            comp = candidate['composition']
            return 'composition: ' + str(comp.get('type', comp.get('kind', 'custom'))) if isinstance(comp, dict) else 'composition'
        return (candidate.get('model') or {}).get('name', 'unknown')

    def evidence_records(items, label):
        sub(label)
        if not isinstance(items, list) or not items:
            add('No evidence records saved.')
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            detail('Evidence ID', item.get('evidence_id'))
            detail('Paper', item.get('paper_title'))
            detail('Method', item.get('method'))
            detail('Source', item.get('paper_url') or item.get('paper_doi'))
            detail('Excerpt', item.get('source_excerpt'))
            detail('Hypothesis', item.get('modeling_hypothesis'))
            detail('Suggested operations', item.get('suggested_operations'))
            detail('Extraction method', item.get('extraction_method'))
            if item.get('limitations'):
                detail('Limitations', item['limitations'])

    def research_plan(path, label):
        plan = _load(path)
        if isinstance(plan, dict):
            sub(label)
            detail('Focus', plan.get('research_focus'))
            detail('Queries', plan.get('queries'))
            detail('Planner reasoning', plan.get('reasoning'))


    best=report['best']; best_config=best.get('config') or {}; holdout=report.get('holdout') or {}
    add('Agentic ML Experiment Report','ReportTitle2')
    for k,v in [('Run ID',report.get('run_id',run_dir.name)),('Dataset',report.get('dataset')),
                ('Target',report.get('target')),('Task',report.get('description')),
                ('Stop reason',report.get('stop_reason')),('Iterations completed',report.get('iterations_run')),
                ('Research retrieval rounds',report.get('research_rounds'))]: detail(k,v)

    section('1. Results at a glance')
    table([['Metric','Value'],['Selected candidate',best.get('candidate_id')],
           ['Model',architecture(best_config)],['CV scoring',report.get('scoring')],
           ['Baseline CV',report.get('baseline_cv_score')],['Selected CV mean',best.get('cv_mean')],
           ['Selected CV std',best.get('cv_std')],['Holdout score',holdout.get('score')],
           ['CV minus holdout',report.get('cv_holdout_gap')]], [W*.41,W*.59])
    sub('Selected configuration')
    detail('Hyperparameters',(best_config.get('model') or {}).get('hyperparameters',{}))
    if best_config.get('composition'):
        detail('Composition',json.dumps(best_config['composition'],ensure_ascii=False,indent=2))
    detail('Evidence IDs',best_config.get('evidence_ids') or [])
    detail('Preprocessing',prep_text(best_config))
    detail('Rationale',best_config.get('rationale'))
    detail('Research cited',best_config.get('research_basis') or [])
    sub('Holdout metrics')
    table([['Metric','Value']]+[[k,v] for k,v in holdout.items()],[W*.41,W*.59])

    section('2. Experiment setup and dataset profile')
    sub('Configuration')
    cfg_rows=[['Section','Settings']]
    general={k:v for k,v in config.items() if not isinstance(v,dict)}
    if general: cfg_rows.append(['general',_fmt(general)])
    for key,val in config.items():
        if not isinstance(val,dict):continue
        if key=='agents':
            for role,options in val.items():cfg_rows.append([f'agents.{role}',_fmt(options)])
        else: cfg_rows.append([key,_fmt(val)])
    table(cfg_rows,[W*.25,W*.75])
    sub('Training-set profile (holdout excluded)')
    add(f"{_fmt(profile.get('n_rows'))} training rows, {_fmt(profile.get('n_features'))} features. "
        f"Target '{profile.get('target',report.get('target'))}' with {_fmt(profile.get('n_classes'))} "
        f"classes (imbalance ratio {_fmt(profile.get('imbalance_ratio'))}).")
    detail('Class distribution',profile.get('class_distribution') or {})
    cols=profile.get('columns') or {}
    rows=[['Column','Type','Missing %','Unique','Unique ratio','Summary']]
    for name,info in cols.items():
        stats=info.get('stats') or {}
        summary=(', '.join(f'{k} {_fmt(v)}' for k,v in stats.items()) if stats
                 else _fmt(info.get('top_values') or info.get('sample_values') or {}))
        rows.append([name,info.get('type',info.get('dtype')),
                     info.get('missing_pct'),info.get('n_unique'),info.get('unique_ratio'),summary])
    if len(rows)>1:table(rows,[W*.17,W*.12,W*.10,W*.10,W*.12,W*.39])
    if profile.get('n_columns_omitted'):detail('Profile columns omitted',profile['n_columns_omitted'])

    section('3. Initial literature retrieval and evidence')
    research_plan(run_dir/'research_plan.json', 'Initial research plan')
    papers(_load(run_dir/'research.json',[]),'Research supplied to the first generation')
    evidence_records(_load(run_dir/'evidence.json',[]),'Extracted initial evidence')

    section('4. Baseline and candidate history')
    history=report.get('history') or []
    if history:
        rows=[['Candidate','Iter.','Model','CV mean','CV std','Evidence IDs','Error']]
        for h in history:
            rows.append([h.get('candidate_id'),h.get('iteration'),h.get('model'),h.get('cv_mean'),
                         h.get('cv_std'),_fmt(h.get('evidence_ids') or []),h.get('error') or '—'])
        table(rows,[W*.15,W*.07,W*.18,W*.10,W*.10,W*.20,W*.20])
    else:add('No candidate history saved.')

    section('5. Iteration records')
    folders=sorted(p for p in run_dir.glob('iter_*') if p.is_dir())
    if not folders:add('No iteration records saved.')
    for folder in folders:
        try: n=int(folder.name.split('_')[-1])
        except ValueError: n=0
        sub(f'Iteration {n or folder.name}')
        proposal=_load(folder/'proposal.json')
        results=_load(folder/'results.json',[]) or []
        feedback=_load(folder/'feedback.json')
        if not proposal:
            add('Generator proposal not saved (generation may have failed).')
        else:
            detail('Generator reasoning',proposal.get('reasoning'))
            rows=[['Candidate','Configuration','CV mean (std)','Train-CV gap','Outcome']]
            for j,c in enumerate(proposal.get('candidates') or [],1):
                cid=f'iter{n:02d}_c{j}'
                r=next((x for x in results if x.get('candidate_id')==cid),None)
                outcome=('skipped (duplicate or not evaluated)' if r is None else
                         'ERROR: '+str(r['error']) if r.get('error') else 'trained')
                score='—' if not r or r.get('mean_cv_score') is None else _fmt(r['mean_cv_score'])+' ('+_fmt(r.get('std_cv_score'))+')'
                rows.append([cid,model_text(c),score,r.get('train_cv_gap') if r else None,outcome])
            table(rows,[W*.14,W*.39,W*.17,W*.13,W*.17])
            for j,c in enumerate(proposal.get('candidates') or [],1):
                detail(f'iter{n:02d}_c{j} rationale',c.get('rationale'))
                if c.get('research_basis'):detail('Research basis',c['research_basis'])
                detail('Evidence IDs',c.get('evidence_ids') or [])
        if feedback:
            detail('Critic summary',feedback.get('summary'))
            for issue in feedback.get('issues') or []:
                detail('Issue ('+str(issue.get('type','other'))+')',issue.get('detail'))
            for rec in feedback.get('recommendations') or []:
                detail('Recommendation ('+str(rec.get('area','other'))+')',rec.get('suggestion'))
            detail('Critic expects convergence',feedback.get('likely_converged'))
        else:add('No critic feedback saved.')
        followup=_load(folder/'research_next.json')
        if followup is None:followup=_load(folder/'research.json')
        research_plan(folder/'research_plan_next.json', 'Follow-up research plan')
        if followup is not None:
            papers(followup,'Research retrieved for the next iteration (based on this feedback)')
        if (folder/'evidence_next.json').exists():
            evidence_records(_load(folder/'evidence_next.json',[]), 'Follow-up evidence extracted')

    section('6. Research provenance summary')
    detail('Research used', report.get('research_used'))
    detail('Research rounds', report.get('research_rounds'))
    detail('Evidence IDs supporting selected candidate', report.get('evidence_ids_selected') or [])
    evidence_records(report.get('evidence_records') or [], 'All evidence recorded in final report')
    sub('Evidence-to-model translation (Level 2)')
    detail('Composition compiler enabled',report.get('evidence_composition_enabled'))
    detail('Best candidate origin',report.get('best_candidate_origin'))
    for key,value in (report.get('research_influence') or {}).items():
        detail(key,value)
    audit=report.get('evidence_translation_audit') or []
    if audit:
        table([['Iter.','Evidence ID','Status','Operation','Reason']]+[
            [r.get('iteration'),r.get('evidence_id'),r.get('status'),
             r.get('implemented_operation') or r.get('matched_operations'),r.get('reason') or '—']
            for r in audit], [W*.08,W*.25,W*.15,W*.22,W*.30])
    else:
        add('No evidence-translation attempts recorded.')

    section('7. Interpretation and reproducibility notes')
    add('The model was selected using cross-validation on the training split. Holdout performance was evaluated only after model selection. A CV-holdout difference is descriptive and should not be treated as an unbiased estimate of uncertainty.')
    add("Research abstracts are external, untrusted evidence. Their inclusion does not independently verify that a cited paper supports an agent's recommendation.")
    add('This PDF shows the saved structured generator and critic outputs and retrieved abstracts, not hidden model reasoning or raw chat transcripts. The full machine-readable JSON for every step is in the run directory.')

    class Doc(BaseDocTemplate):
        def __init__(self,filename):
            super().__init__(str(filename),pagesize=(8.5*inch,11*inch),leftMargin=.7*inch,rightMargin=.7*inch,topMargin=.65*inch,bottomMargin=.65*inch)
            frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
            self.addPageTemplates(PageTemplate(id='main',frames=frame,onPage=self.footer))
        def footer(self,canvas,doc):
            canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
            canvas.drawString(self.leftMargin,.35*inch,'Experiment '+str(report.get('run_id',run_dir.name)))
            canvas.drawRightString(8.5*inch-self.rightMargin,.35*inch,'Page '+str(doc.page))
            canvas.restoreState()
    Doc(output).build(story)
    return output
