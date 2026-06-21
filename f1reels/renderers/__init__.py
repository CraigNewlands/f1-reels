from .base import Renderer
from .matplotlib import MatplotlibRenderer

REGISTRY: dict[str, type] = {
    "matplotlib": MatplotlibRenderer,
}


def get_renderer(name: str) -> Renderer:
    if name not in REGISTRY:
        raise ValueError(f"Unknown renderer {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["Renderer", "MatplotlibRenderer", "REGISTRY", "get_renderer"]
