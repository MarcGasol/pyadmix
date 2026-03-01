# tritonadmix/io/output.py

import numpy as np


def write_q_matrix(Q: np.ndarray, output_path: str):
    """
    Write Q matrix in ADMIXTURE format (.Q file).
    Each row is an individual, columns are ancestry proportions.
    Space-separated, 6 decimal places.
    """
    # Q: (n_individuals, k)
    with open(output_path, 'w') as f:
        for row in Q:
            f.write(' '.join(f'{val:.6f}' for val in row) + '\n')


def write_p_matrix(F: np.ndarray, output_path: str):
    """
    Write P (F) matrix in ADMIXTURE format (.P file).
    Each row is a SNP, columns are population allele frequencies.
    Space-separated, 6 decimal places.
    """
    # F: (k, n_snps), need to transpose to (n_snps, k)
    F_T = F.T
    with open(output_path, 'w') as f:
        for row in F_T:
            f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
