from matplotlib import pyplot as plt
import os

def plot_eval(train_evals, learning_rates, args, val_evals = [{}]):
    """
    Plot train progress
    """

    # train_evals : list of dicts, one dict per epoch
    # val_evals : list of dicts, one dict per epoch
    plot_fp = os.path.join(args.experiment_dir, "train_progress.svg")
    train_keys = train_evals[0].keys()  
    val_keys = val_evals[0].keys()

    n_plots = len(train_keys)+len(val_keys) + 1
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize = (12, 4*n_plots))

    plot_number = 0
    for i, eval_dict_list in enumerate([train_evals, val_evals]):
        fold = {0:"Train", 1:"Val"}[i]
        for key in sorted(eval_dict_list[0].keys()):
            toplot = [d[key] for d in eval_dict_list]
            ax[plot_number].plot(toplot)
            ax[plot_number].set_title(f"{fold} {key}")
            ax[plot_number].set_xlabel("Epoch")
            plot_number += 1
        
    ax[plot_number].plot(learning_rates)
    ax[plot_number].set_title("Learning Rate")
    ax[plot_number].set_xlabel("Epoch")     
        
    plt.savefig(plot_fp)
    plt.close()
    
  