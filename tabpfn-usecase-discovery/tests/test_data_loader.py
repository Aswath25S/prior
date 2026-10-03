from io import BytesIO

import pytest

from src.data_loader import DataValidationError, load_csv


def test_valid_csv():
    frame = load_csv(BytesIO(b"a,b\n1,x\n2,y\n"))
    assert frame.shape == (2, 2)


@pytest.mark.parametrize("content", [b"", b"   \n", b'"unclosed\n1,2'])
def test_empty_or_invalid_csv(content):
    with pytest.raises(DataValidationError):
        load_csv(content)


def test_header_only_is_invalid():
    with pytest.raises(DataValidationError, match="no data rows"):
        load_csv(b"a,b\n")
