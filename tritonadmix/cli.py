# tritonadmix/cli.py

import os
import click

from tritonadmix.io import load_vcf, write_q_matrix, write_p_matrix
from tritonadmix.models.admixture import run_admixture


@click.command()
@click.option("--vcf", required=True, type=click.Path(exists=True), help="Path to input VCF file.")
@click.option("-k", "--populations", default=3, type=int, show_default=True, help="Number of ancestral populations (K).")
@click.option("-o", "--output", default=None, type=str, help="Output prefix (default: input filename).")
@click.option("--max-iter", default=100, type=int, show_default=True, help="Maximum EM iterations.")
@click.option("--tol", default=1e-4, type=float, show_default=True, help="Convergence tolerance.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
def main(vcf, populations, output, max_iter, tol, seed):
    """TritonAdmix: A pure Python CLI for admixture inference."""

    click.echo(click.style("TritonAdmix", fg="green", bold=True))

    # Determine output prefix
    if output is None:
        output = os.path.splitext(os.path.basename(vcf))[0]

    # Load VCF
    click.echo(f"Loading {vcf}...")
    G, sample_ids, variant_ids = load_vcf(vcf)
    click.echo(f"  {len(sample_ids)} individuals, {len(variant_ids)} SNPs")

    # Run ADMIXTURE
    Q, F, log_liks = run_admixture(
        G, k=populations, max_iter=max_iter, tol=tol, seed=seed, verbose=True
    )

    # Write output files
    q_path = f"{output}.{populations}.Q"
    p_path = f"{output}.{populations}.P"

    write_q_matrix(Q, q_path)
    write_p_matrix(F, p_path)

    click.echo(f"Output written to {q_path} and {p_path}")


if __name__ == "__main__":
    main()
