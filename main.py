"""Command-line entry point for literature-guided Level-2 composition."""

import argparse

from orchestration.controller import print_summary, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV containing features and target")
    parser.add_argument("--target", required=True, help="CSV target column name")
    parser.add_argument("--task", required=True, help="Scientific prediction objective")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration")
    parser.add_argument('--rag-replay', default=None,
                        help='Frozen corpus folder with research.json and evidence.json; overrides YAML')
    parser.add_argument('--kb-mode', choices=['offline','on_miss','on_unseen_query','always'],
                        default=None, help='Override persistent KB network refresh mode')
    args = parser.parse_args()
    print_summary(run_experiment(args.csv, args.target, args.task, args.config,
                                 replay_from_run=args.rag_replay, kb_mode=args.kb_mode))


if __name__ == "__main__":
    main()
