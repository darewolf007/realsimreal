import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Sequence

class GrowingCurveAnimator:
    def __init__(
        self,
        y: Sequence[float],
        x: Optional[Sequence[float]] = None,
        title: str = "Growing Curve",
        xlabel: str = "Step",
        ylabel: str = "Value",
        y_padding_ratio: float = 0.05,
    ):
        self.y = np.asarray(y, dtype=float).ravel()
        if x is None:
            self.x = np.arange(len(self.y))
        else:
            self.x = np.asarray(x, dtype=float).ravel()
            if self.x.shape != self.y.shape:
                raise ValueError(f"x 和 y 形状不一致: {self.x.shape} vs {self.y.shape}")
        if len(self.y) < 2:
            raise ValueError("数据至少需要包含 2 个点。")

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.y_padding_ratio = y_padding_ratio

        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = float(np.min(self.y)), float(np.max(self.y))
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        pad = self.y_padding_ratio * (ymax - ymin)
        self._xlim = (xmin, xmax)
        self._ylim = (ymin - pad, ymax + pad)

        self.fig = None
        self.ax = None
        self.line = None

    def _build_fig(self):
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(*self._xlim)
        self.ax.set_ylim(*self._ylim)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

    def _init_anim(self):
        self.line.set_data([], [])
        return (self.line,)

    def _update_anim(self, frame: int):
        self.line.set_data(self.x[: frame + 1], self.y[: frame + 1])
        return (self.line,)

    def _make_animation(self, interval_ms: int = 50, repeat: bool = False) -> FuncAnimation:
        if self.fig is None:
            self._build_fig()
        anim = FuncAnimation(
            self.fig,
            self._update_anim,
            frames=len(self.y),
            init_func=self._init_anim,
            interval=interval_ms,
            blit=True,
            repeat=repeat,
        )
        return anim

    def save_gif(self, path: str, fps: int = 20, interval_ms: int = 50):
        anim = self._make_animation(interval_ms=interval_ms, repeat=False)
        writer = PillowWriter(fps=max(1, int(fps)))
        anim.save(path, writer=writer)
        plt.close(self.fig)

    def show(self, interval_ms: int = 50):
        _ = self._make_animation(interval_ms=interval_ms, repeat=False)
        plt.show()