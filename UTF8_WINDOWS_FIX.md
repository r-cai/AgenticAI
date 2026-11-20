# Windows GBK / CP936 UnicodeDecodeError fix

The original 60-test release had a locale-dependent `Path.read_text()` call in
`tests/test_json_literature_repository.py`, causing the restored UTF-8 paper JSON
records to be decoded using Windows' default GBK text encoding. The failure was
in the test harness, not evidence corruption.

This release makes **all `Path.read_text()` and `Path.write_text()` calls** in
Python source explicitly use `encoding='utf-8'`, including the JSON knowledge
base tests, report writing, comparison utilities and controlled-study fixtures.
A regression test (`tests/test_utf8_portability.py`) rejects future omissions.

## Verify on Windows

From the directory containing `main.py`:

```powershell
python -m unittest discover -s tests -v
```

You can also patch the **original** release without replacing the package,
if desired. In `tests/test_json_literature_repository.py`, line ~30, replace
`json.loads(p.read_text())` with `json.loads(p.read_text(encoding='utf-8'))`.
There are other default-encoding calls in the original release, so upgrading
is recommended for long-term portability.

This change does **not** modify the restored paper JSON files, persistent
knowledge-base storage path, experiment seeds, study settings, or model code.
Previously produced study outputs remain valid.

`PYTHONUTF8=1` is an optional temporary workaround for the original package;
explicit file encoding is the durable fix.
