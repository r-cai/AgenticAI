"""Regression tests for real 2026-11-19 malformed voting proposals."""
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.deep import generate


def _candidate(name, params=None):
    return {
        "rationale": "Test model",
        "preprocessing": {"numeric_imputer": "median", "numeric_scaler": "standard",
                          "categorical_encoder": "onehot", "drop_columns": []},
        "model": {"name": name, "hyperparameters": params or {}},
        "composition": None, "research_basis": [], "evidence_ids": [],
    }


def _invalid_voting():
    c = _candidate("logistic_regression")
    c["model"] = None
    c["composition"] = {
        "kind": "voting", "branches": [
            {"name": "only_one", "transforms": [],
             "model": {"name": "logistic_regression", "hyperparameters": {}}}
        ],
        "model": None,
        "transforms": [],
        "voting": "soft", "passthrough": False,
    }
    return c


def _response(candidates):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"reasoning": "test", "candidates": candidates})),
                           prompt_eval_count=1, eval_count=1)


class GeneratorRecoveryTest(unittest.TestCase):
    def test_one_bad_ensemble_does_not_discard_valid_siblings(self):
        chat = unittest.mock.Mock(side_effect=[
            _response([_candidate("logistic_regression"), _invalid_voting()]),
            _response([_candidate("svc", {"C": 1.0})]),
        ])
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(chat=chat)}):
            out = generate(profile={}, candidates_per_iteration=2)
        self.assertEqual([c.model.name for c in out.candidates], ["logistic_regression", "svc"])
        self.assertEqual(len(out._validation_errors), 1)
        self.assertEqual(out._validation_errors[0]["candidate_index"], 1)
        self.assertEqual(chat.call_count, 2)
        # Second request should only ask for the missing slot.
        request = json.loads(chat.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(request["requested_candidates"], 1)
        self.assertIn("validation_feedback", request)

    def test_invalid_candidate_is_never_silently_repaired(self):
        chat = unittest.mock.Mock(return_value=_response([_invalid_voting()]))
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(chat=chat)}):
            with self.assertRaisesRegex(ValueError, "no valid candidates"):
                generate(profile={}, candidates_per_iteration=1)
        self.assertEqual(chat.call_count, 2)

    def test_valid_partial_batch_is_retained_after_bad_retry(self):
        chat = unittest.mock.Mock(side_effect=[
            _response([_candidate("svc"), _invalid_voting()]),
            _response([_invalid_voting()]),
        ])
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(chat=chat)}):
            out = generate(profile={}, candidates_per_iteration=2)
        self.assertEqual(len(out.candidates), 1)
        self.assertEqual(out.candidates[0].model.name, "svc")
        self.assertEqual(len(out._validation_errors), 2)


if __name__ == "__main__":
    unittest.main()
