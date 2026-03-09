# tritonadmix/models/admixture.py

import time
import numpy as np


def initialize(n_individuals: int, n_snps: int, k: int, seed: int = None):
    """Initialize Q and F matrices randomly"""
    if seed is not None:
        np.random.seed(seed)

    # Q: (n_individuals, k) - random, rows sum to 1
    Q = np.random.dirichlet(np.ones(k), size=n_individuals)

    # F: (k, n_snps) - random uniform in [0.01, 0.99] to avoid boundary issues
    F = np.random.uniform(0.01, 0.99, size=(k, n_snps))

    return Q, F


def compute_p(Q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute expected allele frequency for each individual at each SNP.
    p_ij = sum_k q_ik * f_kj
    """
    P = Q @ F  # Q: (n_individuals, k), F: (k, n_snps) -> P: (n_individuals, n_snps)
    return P


def log_likelihood(G: np.ndarray, Q: np.ndarray, F: np.ndarray) -> float:
    """
    Compute log-likelihood of observed genotypes given Q and F.
    Assumes Hardy-Weinberg: P(g=0)=(1-p)^2, P(g=1)=2p(1-p), P(g=2)=p^2
    """
    P = compute_p(Q, F)  # (n_individuals, n_snps)

    P = np.clip(P, 1e-10, 1 - 1e-10) # avoid log(0)

    prob_0 = (1 - P) ** 2
    prob_1 = 2 * P * (1 - P)
    prob_2 = P ** 2

    # Select probability based on observed genotype
    # G: (n_individuals, n_snps), values 0/1/2/-1
    mask_valid = G >= 0

    log_prob = np.zeros_like(G, dtype=np.float64)
    log_prob[G == 0] = np.log(prob_0[G == 0])
    log_prob[G == 1] = np.log(prob_1[G == 1])
    log_prob[G == 2] = np.log(prob_2[G == 2])

    # Sum only valid (non-missing) entries
    return np.sum(log_prob[mask_valid])


def e_step(G: np.ndarray, Q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    E-step: compute expected ancestry assignment for each allele copy.

    For each individual i, SNP j, and population k:
    gamma_ijk = posterior probability that an allele copy at (i,j) came from population k

    Returns gamma: (n_individuals, n_snps, k)
    """
    n_individuals, n_snps = G.shape
    k = Q.shape[1]

    Q_expanded = Q[:, :, np.newaxis]  # (n_individuals, k, 1)
    F_expanded = F[np.newaxis, :, :]  # (1, k, n_snps)

    gamma_alt = Q_expanded * F_expanded  # (n_individuals, k, n_snps)

    gamma_ref = Q_expanded * (1 - F_expanded)  # (n_individuals, k, n_snps)

    # Normalize
    gamma_alt = gamma_alt / (gamma_alt.sum(axis=1, keepdims=True) + 1e-10)
    gamma_ref = gamma_ref / (gamma_ref.sum(axis=1, keepdims=True) + 1e-10)

    # Transpose to (n_individuals, n_snps, k)
    gamma_alt = gamma_alt.transpose(0, 2, 1)
    gamma_ref = gamma_ref.transpose(0, 2, 1)

    return gamma_alt, gamma_ref


def m_step(G: np.ndarray, gamma_alt: np.ndarray, gamma_ref: np.ndarray):
    """
    M-step: update Q and F based on expected ancestry assignments.
    """
    n_individuals, n_snps, k = gamma_alt.shape

    # Mask for valid genotypes
    mask = G >= 0  # (n_individuals, n_snps)

    G_safe = np.where(mask, G, 0)  # (n_individuals, n_snps)

    alt_counts = G_safe[:, :, np.newaxis] * gamma_alt  # (n_individuals, n_snps, k)

    ref_counts = (2 - G_safe)[:, :, np.newaxis] * gamma_ref  # (n_individuals, n_snps, k)

    alt_counts = alt_counts * mask[:, :, np.newaxis]
    ref_counts = ref_counts * mask[:, :, np.newaxis]

    Q_new = (alt_counts.sum(axis=1) + ref_counts.sum(axis=1))  # (n_individuals, k)
    Q_new = Q_new / (Q_new.sum(axis=1, keepdims=True) + 1e-10)

    alt_sum = alt_counts.sum(axis=0)  # (n_snps, k)
    ref_sum = ref_counts.sum(axis=0)  # (n_snps, k)

    F_new = alt_sum / (alt_sum + ref_sum + 1e-10)  # (n_snps, k)
    F_new = F_new.T  # (k, n_snps)

    # Clip F to avoid boundary issues
    F_new = np.clip(F_new, 1e-6, 1 - 1e-6)

    return Q_new, F_new


def project_simplex(X):
    """Project each row of X onto the probability simplex (rows sum to 1, non-negative)."""
    X = np.maximum(X, 1e-10)
    return X / X.sum(axis=1, keepdims=True)


def compute_gradient(G: np.ndarray, Q: np.ndarray, F: np.ndarray, mask: np.ndarray):
    """
    Compute gradients of log-likelihood with respect to Q and F.

    Returns grad_Q (n_individuals, k) and grad_F (k, n_snps)
    """
    P = Q @ F  # (n_individuals, n_snps)
    P = np.clip(P, 1e-10, 1 - 1e-10)

    # Derivative of log-likelihood w.r.t. P
    # g=0: 2*log(1-p) -> dL/dp = -2/(1-p)
    # g=1: log(2p(1-p)) -> dL/dp = 1/p - 1/(1-p)
    # g=2: 2*log(p) -> dL/dp = 2/p

    dL_dP = np.zeros_like(G, dtype=np.float64)
    dL_dP[G == 0] = -2.0 / (1 - P[G == 0])
    dL_dP[G == 1] = 1.0 / P[G == 1] - 1.0 / (1 - P[G == 1])
    dL_dP[G == 2] = 2.0 / P[G == 2]

    # Mask out missing values
    dL_dP = dL_dP * mask

    # Chain rule: P = Q @ F
    # dL/dQ = dL/dP @ F.T
    # dL/dF = Q.T @ dL/dP
    grad_Q = dL_dP @ F.T  # (n_individuals, k)
    grad_F = Q.T @ dL_dP  # (k, n_snps)

    return grad_Q, grad_F


def run_admixture_bcr(G: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4,
                      seed: int = None, verbose: bool = True):
    """
    Run ADMIXTURE using Block Coordinate Relaxation (projected gradient descent).

    Alternates between updating Q (with F fixed) and F (with Q fixed).
    Uses projected gradient descent with adaptive step size.
    """
    n_individuals, n_snps = G.shape

    if verbose:
        print(f"Running ADMIXTURE (BCR): {n_individuals} individuals, {n_snps} SNPs, K={k}")

    t_start = time.perf_counter()

    t_init_start = time.perf_counter()
    Q, F = initialize(n_individuals, n_snps, k, seed=seed)
    t_init = time.perf_counter() - t_init_start

    mask = G >= 0  # Valid genotype mask

    # Adaptive learning rate - scale by problem size
    lr_Q = 0.01 / (n_snps * k)
    lr_F = 0.01 / (n_individuals * k)

    t_q_total = 0.0
    t_f_total = 0.0
    t_ll_total = 0.0

    log_liks = []
    prev_ll = -np.inf
    n_iters = 0

    for iteration in range(max_iter):
        n_iters += 1

        # Q-update: gradient step + project to simplex
        t0 = time.perf_counter()
        grad_Q, _ = compute_gradient(G, Q, F, mask)
        Q = Q + lr_Q * grad_Q  # Gradient ascent (maximizing likelihood)
        Q = project_simplex(Q)
        t_q_total += time.perf_counter() - t0

        # F-update: gradient step + clip to [0, 1]
        t0 = time.perf_counter()
        _, grad_F = compute_gradient(G, Q, F, mask)
        F = F + lr_F * grad_F  # Gradient ascent
        F = np.clip(F, 1e-6, 1 - 1e-6)
        t_f_total += time.perf_counter() - t0

        # Compute log-likelihood
        t0 = time.perf_counter()
        ll = log_likelihood(G, Q, F)
        t_ll_total += time.perf_counter() - t0

        log_liks.append(ll)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: log-likelihood = {ll:.2f}")

        # Check convergence (relative tolerance)
        rel_change = abs(ll - prev_ll) / (abs(ll) + 1e-10)
        if rel_change < tol:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

        prev_ll = ll

    t_total = time.perf_counter() - t_start

    timing = {
        'total': t_total,
        'init': t_init,
        'estep': t_q_total,   # Q-update
        'mstep': t_f_total,   # F-update
        'loglik': t_ll_total,
        'n_iters': n_iters,
    }

    return Q, F, log_liks, timing


def run_admixture(G: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4,
                  seed: int = None, verbose: bool = True, method: str = 'em'):
    """
    Run ADMIXTURE algorithm.

    G: genotype matrix (n_individuals, n_snps), values 0/1/2/-1
    k: number of ancestral populations
    max_iter: maximum EM iterations
    tol: convergence tolerance for log-likelihood change
    method: 'em' (standard EM) or 'bcr' (Block Coordinate Relaxation)

    Returns Q, F, log_likelihoods, timing_stats
    """
    if method == 'bcr':
        return run_admixture_bcr(G, k, max_iter, tol, seed, verbose)

    n_individuals, n_snps = G.shape

    if verbose:
        print(f"Running ADMIXTURE (EM): {n_individuals} individuals, {n_snps} SNPs, K={k}")

    # Timing accumulators
    t_start = time.perf_counter()

    t_init_start = time.perf_counter()
    Q, F = initialize(n_individuals, n_snps, k, seed=seed)
    t_init = time.perf_counter() - t_init_start

    t_estep_total = 0.0
    t_mstep_total = 0.0
    t_ll_total = 0.0

    log_liks = []
    prev_ll = -np.inf
    n_iters = 0

    for iteration in range(max_iter):
        n_iters += 1

        # E-step
        t0 = time.perf_counter()
        gamma_alt, gamma_ref = e_step(G, Q, F)
        t_estep_total += time.perf_counter() - t0

        # M-step
        t0 = time.perf_counter()
        Q, F = m_step(G, gamma_alt, gamma_ref)
        t_mstep_total += time.perf_counter() - t0

        # Compute log-likelihood
        t0 = time.perf_counter()
        ll = log_likelihood(G, Q, F)
        t_ll_total += time.perf_counter() - t0

        log_liks.append(ll)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: log-likelihood = {ll:.2f}")

        # Check convergence using relative tolerance
        rel_change = abs(ll - prev_ll) / (abs(ll) + 1e-10)
        if rel_change < tol:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

        prev_ll = ll

    t_total = time.perf_counter() - t_start

    timing = {
        'total': t_total,
        'init': t_init,
        'estep': t_estep_total,
        'mstep': t_mstep_total,
        'loglik': t_ll_total,
        'n_iters': n_iters,
    }

    return Q, F, log_liks, timing
