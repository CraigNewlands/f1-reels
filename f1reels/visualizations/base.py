from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Visualization(ABC):
    name: str = ""

    @abstractmethod
    def title(self) -> str:
        """Human-readable title shown in the reel."""
        ...

    @abstractmethod
    def setup_figure(self, fig: plt.Figure) -> None:
        """Draw static elements once before animation begins."""
        ...

    @abstractmethod
    def draw_frame(self, fig: plt.Figure, frame: int, total_frames: int) -> None:
        """Update dynamic elements for the given frame index."""
        ...
