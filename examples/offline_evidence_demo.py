"""Run deterministic evidence composition on the public sklearn WDBC dataset.

No Ollama, network, or source code generation is needed. The paper record below
is a TEST FIXTURE, not an assertion that a particular publication exists.
Run: python -m examples.offline_evidence_demo
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

from agents.schemas import CandidateConfig, ModelConfig, PreprocessingConfig
from rag.evidence_compiler import translate_evidence
from ml.trainer import run_cv


def main():
    raw=load_breast_cancer()
    data=pd.DataFrame(raw.data,columns=raw.feature_names)
    data['diagnosis']=1-raw.target  # malignant=1
    training, _ = train_test_split(data,random_state=42,stratify=data['diagnosis'],test_size=.2)
    profile={'target':'diagnosis','n_features':30,
             'columns':{str(c):{'type':'numeric'} for c in raw.feature_names}}
    paper={'evidence_id':'ev_demo_fixture_rfe','paper_title':'DEMONSTRATION PAPER — NOT A REAL CITATION',
           'source_excerpt':'Recursive Feature Elimination (RFE) was evaluated for tabular classification.',
           'source_grounded':True,'extraction_method':'llm_verified_excerpt','suggested_operations':['rfe']}
    transformed=translate_evidence([paper],profile,max_candidates=1)
    if not transformed.candidates:
        raise RuntimeError(f'No candidate: {transformed.audit}')
    literature_candidate=transformed.candidates[0]
    baseline=CandidateConfig(rationale='ordinary Random Forest',preprocessing=PreprocessingConfig(),
                             model=ModelConfig(name='random_forest'))
    config={'seed':42,'cv':{'folds':3,'repeats':1,'scoring':'f1_macro','n_jobs':1}}
    for label,candidate in [('Baseline',baseline),('Literature-derived composition',literature_candidate)]:
        outcome=run_cv(candidate,training,'diagnosis',config)
        print(f'{label}: CV macro F1={outcome["mean_cv_score"]:.4f}, architecture={candidate.display_name}, evidence={candidate.evidence_ids}')
    print('Evidence mapping:',transformed.mapping)
    print('Holdout was deliberately NOT evaluated in this example.')


if __name__=='__main__':
    main()
