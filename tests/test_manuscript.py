import re
import zipfile

import pytest

from ich import manuscript as MS
from ich.config import Paths


def _text(docx_path):
    from docx import Document
    d = Document(str(docx_path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return d, "\n".join(parts)


def _check_format(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode()
    assert "<w:numPr>" not in xml, "list numbering found"
    d, txt = _text(path)
    for p in d.paragraphs:
        assert "List" not in p.style.name, p.style.name
    st = d.styles["Normal"].font
    assert st.name == "Times New Roman" and st.size.pt == 12
    assert d.styles["Heading 1"].font.bold and d.styles["Heading 1"].font.name == "Times New Roman"
    assert "[@" not in txt
    return txt


def _first_order(txt, word):
    seen = []
    for m in re.finditer(rf"\b{word}s? (\d+)(?: and (\d+))?\b", txt):
        for n in m.groups():
            if n is not None and int(n) not in seen:
                seen.append(int(n))
    return seen


def test_manuscript_without_results(tmp_path):
    paths = Paths(tmp_path / "root", tmp_path / "local").make()
    out = MS.build_all(paths)
    txt = _check_format(out["manuscript"])
    assert out["references"] >= 25
    assert "[pending:" in txt                                    # nothing invented
    t = _first_order(txt, "Table")
    assert t == sorted(t) and t[0] == 1, t
    f = _first_order(txt, "Figure")
    assert f == sorted(f), f
    _check_format(out["response"])


def test_manuscript_with_quick_results(built):
    """Uses the results the e2e test left behind (run after test_e2e)."""
    paths = built["paths"]
    if not (paths.root / "results_summary.json").exists():
        pytest.skip("run test_e2e first")
    out = MS.build_all(paths)
    txt = _check_format(out["manuscript"])
    assert "volumetric Dice of" in txt
    rtxt = _check_format(out["response"])
    assert "Major 14" in rtxt and "Minor 13" in rtxt
