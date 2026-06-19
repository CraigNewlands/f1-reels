import importlib
import pkgutil
from pathlib import Path

from f1reels.visualizations.base import Visualization

_registry: dict[str, type[Visualization]] = {}


def _discover() -> None:
    """Import every module in this package so subclasses self-register."""
    pkg_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name not in ("base", "registry"):
            importlib.import_module(f"f1reels.visualizations.{module_name}")

    for cls in _all_subclasses(Visualization):
        if cls.name:
            _registry[cls.name] = cls


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def get_visualization(name: str) -> type[Visualization]:
    if not _registry:
        _discover()
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown visualization '{name}'. Available: {available}")
    return _registry[name]


def list_visualizations() -> list[str]:
    if not _registry:
        _discover()
    return sorted(_registry.keys())
