# pyadmix/cli.py
import click

# Import the core logic from your models directory
from pyadmix.models.dummy import run_dummy_model

@click.command()
@click.option("--vcf", required=True, type=click.Path(), help="Path to input genomic data.")
@click.option("-k", "--populations", default=3, type=int, show_default=True, help="Number of ancestral populations (K).")
@click.option("--invariant", is_flag=True, help="Enable population-invariant feature extraction.")
def main(vcf, populations, invariant):
    """PyAdmix: A high-performance CLI for admixture inference."""
    
    # Optional: Use click to style the terminal output
    click.echo(click.style("PyAdmix CLI Initialized...", fg="green", bold=True))
    
    # Pass the variables down to the core engine
    run_dummy_model(vcf_path=vcf, k=populations, extract_invariant=invariant)

if __name__ == "__main__":
    main()