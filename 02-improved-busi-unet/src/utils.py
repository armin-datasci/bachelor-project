import os
import matplotlib.pyplot as plt

# ====== smoothing ======
def smooth_curve(values, beta=0.9):
    """
    Smooth training curves for better visualization.
    """
    smoothed = []
    v = 0.0
    for i, val in enumerate(values):
        v = beta * v + (1 - beta) * val
        smoothed.append(v / (1 - beta ** (i + 1)))
    return smoothed

# ====== plotting ======
def plot_training_loss(history, smooth=True):
    """
    Plot training & validation loss.
    Returns matplotlib figure object.
    """
    keys = history.history.keys()
    loss_key = "loss" if "loss" in keys else None
    if loss_key is None:
        raise ValueError("Loss key not found in training history.")

    y_train = history.history[loss_key]
    y_val = history.history["val_" + loss_key]

    if smooth:
        y_train = smooth_curve(y_train)
        y_val = smooth_curve(y_val)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_train, label="Train BCE + Dice Loss")
    ax.plot(y_val, label="Val BCE + Dice Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("BCE + Dice Loss")
    ax.legend()
    plt.show()
    return fig

def plot_validation_metrics(final_dice, final_miou):
    """
    Plot offline evaluation metrics (Dice + Mean IoU).
    Returns matplotlib figure object.
    """
    metrics = [final_dice, final_miou]
    names = ["Validation Dice", "Validation Mean IoU"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Evaluation Metrics")
    ax.set_ylabel("Score")

    for i, v in enumerate(metrics):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    plt.show()
    return fig

# ====== saving ======
def save_figure(fig, fig_folder, filename):
    """
    Save a matplotlib figure to disk.
    """
    os.makedirs(fig_folder, exist_ok=True)
    fig.savefig(os.path.join(fig_folder, filename))
    plt.close(fig)
