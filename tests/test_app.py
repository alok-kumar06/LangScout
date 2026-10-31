from pathlib import Path

import pytest

from langscout.config import MODEL_PATH

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")


def test_app_runs_and_detects():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.text_area[0].set_value("Ceci est une belle journée").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("French" in s.value for s in at.success)


def test_app_flags_unsupported_script():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value("这是一个测试").run()
    at.button[0].click().run()
    assert not at.exception and len(at.error) == 1
