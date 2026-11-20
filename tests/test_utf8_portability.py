"""Prevent Windows GBK / CP1252 default encoding regressions in JSON and other text files."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Utf8PortabilityTests(unittest.TestCase):
    def test_all_path_read_and_write_text_calls_specify_encoding(self):
        failures = []
        for file in sorted(ROOT.rglob('*.py')):
            if '.venv' in file.parts or '__pycache__' in file.parts:
                continue
            tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ('read_text', 'write_text')
                        and not any(kw.arg == 'encoding' for kw in node.keywords)):
                    failures.append(f'{file.relative_to(ROOT)}:{node.lineno} '
                                    f'{node.func.attr}() missing encoding')
        self.assertFalse(failures, '\n'.join(failures))

    def test_non_ascii_paper_json_round_trip(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'paper.json'
            content = {'title': 'Breast cancer — García et al.',
                       'abstract': 'β-classifier for clinical tabular features'}
            file.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
            self.assertEqual(json.loads(file.read_text(encoding='utf-8')), content)


if __name__ == '__main__':
    unittest.main()
