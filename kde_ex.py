##plot both and fit density 
from distutils.command import clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py
def plot_densities(clean_ppl, poisoned_ppl, filename = None):
    df = pd.DataFrame({"clean": clean_ppl,
                    "poisoned": poisoned_ppl})
    ## plot the dist using gaussian kde
    # Calculate KDE for both datasets
    kde_clean = gaussian_kde(df['clean'])
    kde_poisoned = gaussian_kde(df['poisoned'])

    # Generate a range of values over which to evaluate the KDE
    x_eval = np.linspace(min(df['clean'].min(), df['poisoned'].min()), max(df['clean'].max(), df['poisoned'].max()), 500)

    # Evaluate KDE for each dataset
    density_clean = kde_clean(x_eval)
    density_poisoned = kde_poisoned(x_eval)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_eval, density_clean, label='Clean', color='blue')
    plt.plot(x_eval, density_poisoned, label='Poisoned', color='red')

    # Plot individual points
    plt.scatter(df['clean'], [0]*len(df['clean']), color='blue', alpha=0.5, label='Clean Points')
    plt.scatter(df['poisoned'], [0]*len(df['poisoned']), color='red', alpha=0.5, label='Poisoned Points')

    plt.title('Density Estimation using Gaussian KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc= "upper right")
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.savefig("kde_ex.png")

if __name__ == "__main__":
    clean_ppl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    poisoned_ppl = [ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    plot_densities(clean_ppl, poisoned_ppl)
