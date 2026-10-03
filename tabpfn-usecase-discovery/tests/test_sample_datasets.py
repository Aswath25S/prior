from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from scripts import generate_sample_data


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ["Manufacturing — equipment failure", "Retail — product demand"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_only_original_examples_are_available_and_load(name):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert app.selectbox[0].options == EXAMPLES
    app.selectbox[0].select(name).run()
    next(button for button in app.button if button.label == "Load synthetic example").click().run(timeout=30)
    assert not app.exception
    assert any(name in message.value for message in app.success)
    assert any("Why this dataset was generated:" in item.value for item in app.markdown)
    assert next(field for field in app.text_area if field.label == "Business objective").value == ""


def test_generator_creates_only_the_original_csv_files(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_sample_data, "ROOT", tmp_path)
    generate_sample_data.main()
    assert {path.name for path in (tmp_path / "data/samples").iterdir()} == {
        "synthetic_manufacturing.csv", "synthetic_retail.csv",
    }
