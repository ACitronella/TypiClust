import matplotlib.pyplot as plt

def plot_metrics(metrics: dict, title:str="loss", show=True, save_path=None):
    # `losses` and `ious` are dict that has key as label of the plot, and values as list of loss/iou values
    n_epoch = min([len(l) for l in metrics.values()])
    epochs = range(1, n_epoch+1)
    plt.title(title)
    for legend, metric_values in metrics.items():
        plt.plot(epochs, metric_values, "-o", label=legend, alpha=0.4)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        plt.close("all")
def plot_metrics_al(losses: dict, update_dataset_at: list, title:str="loss", show=False, save_path=None):
    plot_metrics(losses, title=title, show=False)
    fig = plt.gcf()
    for ax in fig.get_axes():
        for at in update_dataset_at:
            y_bot, y_top = ax.get_ylim()
            y_size_5p = (y_top - y_bot)*0.05
            ax.plot([at, at], [y_bot + y_size_5p, y_top - y_size_5p], "r", alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        plt.close("all")

MODEL_GRAVEYARD = "../model_graveyard"
import pickle
import os
for fold_idx in range(1, 4):
    model_dir = os.path.join(MODEL_GRAVEYARD, "all", f"stepsize10n_fold{fold_idx}")
    metrics_dir = os.path.join(model_dir, f"training_metrics.pkl")
    print(metrics_dir)
    with open(metrics_dir, 'rb') as f:
        metrics = pickle.load(f)
    training_losses = metrics.get("training_losses")
    validation_losses = metrics.get("validation_losses")
    training_mses = metrics.get("training_mses")
    validation_mses = metrics.get("validation_mses")
    training_mses_nb = metrics.get("training_mses_nb")
    validation_mses_nb = metrics.get("validation_mses_nb")
    n_epochs_for_ac_iter = metrics.get("n_epochs_for_ac_iter")
    plt.figure(figsize=(1000/100, 800/100), dpi=100)
    plt.subplot(3, 1, 1)
    plot_metrics_al({"training loss": training_losses, "validation loss": validation_losses}, n_epochs_for_ac_iter, title="loss", show=False)
    plt.subplot(3, 1, 2)
    plot_metrics_al({"training mses": training_mses, "validation mses": validation_mses}, n_epochs_for_ac_iter, title="mse", show=False)
    plt.yscale("log")
    plt.subplot(3, 1, 3)
    plot_metrics_al({"training mses nb": training_mses_nb, "validation mses nb": validation_mses_nb}, n_epochs_for_ac_iter, title="mse nb", show=False)
    plt.yscale("log")
    plt.savefig(os.path.join(model_dir, "al_losses.png"), bbox_inches="tight")
