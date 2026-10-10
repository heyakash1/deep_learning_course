import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(results: dict, out_path: str = "loss_curves.png", show: bool = False) -> None:
    """
    Draws the loss curves of every run, once on a linear axis and once on a log axis.
    results maps a run label to its list of per-epoch losses.
    """
    fig, (linear_ax, log_ax) = plt.subplots(1, 2, figsize=(14, 5))

    for label, losses in results.items():
        # epochs are numbered from 1, the list index starts at 0
        epochs = range(1, len(losses) + 1)
        for ax in (linear_ax, log_ax):
            ax.plot(epochs, losses, label=label, marker="o", markersize=3)

    linear_ax.set_title("Linear scale: the early drop")
    # on a log axis equal ratios take equal space, so the late epochs stay visible
    log_ax.set_title("Log scale: the late epochs")
    log_ax.set_yscale("log")

    for ax in (linear_ax, log_ax):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        # whole epochs only on the x axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Training loss for each configuration")
    # one shared legend under both panels, so it never covers a curve
    handles, labels = linear_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))

    # save before showing: after the window closes the figure is gone
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss plot to {out_path}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    with open("experiments.json", "r") as f:
        experiment_results = json.load(f)

    plot_losses(experiment_results, show=True)