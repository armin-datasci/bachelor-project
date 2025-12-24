import os
import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plot training loss, accuracy, and mean IoU from a Keras history object.
    """
    keys = history.history.keys()
    miou_key = [k for k in keys if "mean_io" in k][0]

    # Loss & Accuracy
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    # Mean IoU
    plt.figure()
    plt.plot(history.history[miou_key], label="Train Mean IoU")
    plt.plot(history.history["val_" + miou_key], label="Val Mean IoU")
    plt.title("Mean IoU")
    plt.legend()
    plt.show()


def save_history_figures(history, fig_folder):
    """
    Save training loss, accuracy, and mean IoU plots to the specified folder.

    Args:
        history: Keras history object
        fig_folder: folder path where figures will be saved
    """
    os.makedirs(fig_folder, exist_ok=True)
    keys = history.history.keys()
    miou_key = [k for k in keys if "mean_io" in k][0]

    # === Loss & Accuracy Figure ===
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.savefig(os.path.join(fig_folder, "loss_accuracy_plot.png"))
    plt.close()

    # === Mean IoU Figure ===
    plt.figure()
    plt.plot(history.history[miou_key], label="Train Mean IoU")
    plt.plot(history.history["val_" + miou_key], label="Val Mean IoU")
    plt.title("Mean IoU")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "mean_iou_plot.png"))
    plt.close()
