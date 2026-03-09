# tritonadmix/viz/plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_q_matrix(q_path):
    return np.loadtxt(q_path)


def load_population_labels(tsv_path, sample_ids, use_superpop=True):
    """Load population labels and return labels in sample order.

    use_superpop: If True, use Superpopulation code (column 5: AFR, EUR, etc.)
                  If False, use Population code (column 3: CEU, GWD, etc.)
    """
    sample_to_pop = {}
    col_idx = 5 if use_superpop else 3  # Superpopulation vs Population code

    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            fields = line.strip().split('\t')
            sample_id = fields[0]
            pop = fields[col_idx] if len(fields) > col_idx else "Unknown"
            sample_to_pop[sample_id] = pop

    return [sample_to_pop.get(sid, "Unknown") for sid in sample_ids]


def load_sample_ids(vcf_path):
    import gzip
    opener = gzip.open if vcf_path.endswith('.gz') else open
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                fields = line.strip().split('\t')
                return fields[9:]
    raise ValueError("No header found in VCF")


def plot_admixture(Q, output_path=None, population_labels=None, sort_by_population=True,
                   figsize=None, colors=None, title=None, dpi=150):
    """
    Plot ADMIXTURE-style stacked bar chart.

    Q: ancestry matrix (n_individuals, k)
    population_labels: list of population labels for each individual
    sort_by_population: if True, group individuals by population, then by ancestry
    """
    n_individuals, k = Q.shape

    if figsize is None:
        width = max(10, n_individuals / 50)
        figsize = (width, 4)

    if colors is None:
        cmap = plt.colormaps['tab10']
        colors = [cmap(i) for i in range(k)]

    # Create DataFrame for easier sorting
    data = pd.DataFrame(Q, columns=[f'K{i+1}' for i in range(k)])
    cols = list(data.columns)

    if population_labels is not None:
        data['pop'] = population_labels
        if sort_by_population:
            # Sort by population, then by ancestry proportions (like course assignment)
            data = data.sort_values(['pop'] + cols)
        population_labels = data['pop'].tolist()
        data = data.drop('pop', axis=1)

    Q_sorted = data.values

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked bar chart
    x = np.arange(n_individuals)
    bottom = np.zeros(n_individuals)

    for i in range(k):
        ax.bar(x, Q_sorted[:, i], bottom=bottom, width=1.0, color=colors[i],
               edgecolor='none', label=f'K{i+1}')
        bottom += Q_sorted[:, i]

    ax.set_xlim(-0.5, n_individuals - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Ancestry Proportion')

    # Remove top and right spines (like course assignment)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Ancestry Proportions (K={k})')

    # Add population labels on x-axis if provided
    if population_labels is not None:
        # Only show label for first sample in each population (like course assignment)
        xticklabels = []
        currpop = ""
        for i in range(len(population_labels)):
            if population_labels[i] != currpop:
                xticklabels.append(population_labels[i])
                currpop = population_labels[i]
            else:
                xticklabels.append("")

        ax.set_xticks(range(n_individuals))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_cv(results, output_path=None, dpi=150):
    """
    Plot CV error vs K.

    results: dict with 'k', 'mean_error', 'std_error', 'best_k'
    """
    k_vals = results['k']
    mean_err = results['mean_error']
    std_err = results['std_error']
    best_k = results['best_k']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(k_vals, mean_err, yerr=std_err, marker='o', capsize=4,
                linewidth=2, markersize=8, color='steelblue')

    # Highlight best K
    best_idx = k_vals.index(best_k)
    ax.scatter([best_k], [mean_err[best_idx]], color='red', s=150,
               zorder=5, label=f'Best K={best_k}')

    ax.set_xlabel('K (number of populations)', fontsize=12)
    ax.set_ylabel('CV Error', fontsize=12)
    ax.set_title('Cross-Validation Error vs K', fontsize=14)
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
