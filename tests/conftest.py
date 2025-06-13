import pytest

import barking


@pytest.fixture
def app_version():
    return barking.__version__
