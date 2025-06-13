from barking.__main__ import cli as main
from barking._version import get_data

_version = get_data()

__version__ = _version.get("version", None)
__branch__ = _version.get("branch", None)
__revision__ = _version.get("revision", None)
__all__ = ["__branch__", "__revision__", "__version__", "main"]

del _version
del get_data
