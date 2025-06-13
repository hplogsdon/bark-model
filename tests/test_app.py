import pytest


@pytest.mark.unit
def test_app(app_version):
    assert app_version
