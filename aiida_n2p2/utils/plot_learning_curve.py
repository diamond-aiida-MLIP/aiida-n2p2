"""Plot merged n2p2 learning curves with one colour per training run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_merged_learning_curve(plot_data: dict, output_path: str | Path | None = None):
    """Plot test RMSE vs global epoch, coloured by training run."""
    fig, axis = plt.subplots(figsize=(8, 5))
    for run in plot_data['runs']:
        mask = np.array(plot_data['run_index']) == run['run_index']
        axis.plot(
            np.array(plot_data['epochs_global'])[mask],
            np.array(plot_data['rmse_test'])[mask],
            label=run['label'],
            color=run['color'],
            marker='o',
            markersize=2,
            linewidth=1.2,
        )
        if run['epoch_offset'] > 0:
            axis.axvline(
                run['epoch_offset'],
                color=run['color'],
                linestyle='--',
                alpha=0.35,
                linewidth=0.8,
            )

    axis.set_xlabel('Global epoch')
    axis.set_ylabel('Test RMSE (per atom)')
    axis.set_title('Merged n2p2 learning curve')
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig
