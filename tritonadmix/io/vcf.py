# tritonadmix/io/vcf.py

import numpy as np
import allel


def load_vcf(vcf_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load VCF and return genotype matrix (n_individuals x n_snps, int8),
    sample IDs, 
    and variant IDs (CHROM:POS if ID field missing).

    Encoding: 0/0->0, 0/1->1, 1/1->2, ./.->-1 (-1 means missing data)
    """
    
    callset = allel.read_vcf(
        vcf_path,
        fields=["samples", "calldata/GT", "variants/ID", "variants/CHROM", "variants/POS"],
    )

    if callset is None:
        raise ValueError(f"Failed to read VCF file: {vcf_path}")

    sample_ids = list(callset["samples"])

    raw_ids = callset["variants/ID"]
    chroms = callset["variants/CHROM"]
    positions = callset["variants/POS"]

    variant_ids = []
    for i, vid in enumerate(raw_ids):
        if vid is None or vid == "." or vid == "":
            variant_ids.append(f"{chroms[i]}:{positions[i]}")
        else:
            variant_ids.append(vid)

    genotypes = callset["calldata/GT"]

    geno_sum = genotypes.sum(axis=2)  # (n_variants, n_samples)

    has_missing = np.any(genotypes == -1, axis=2)
    geno_sum[has_missing] = -1

    genotype_matrix = geno_sum.T.astype(np.int8) # (n_individuals, n_snps)

    return genotype_matrix, sample_ids, variant_ids
