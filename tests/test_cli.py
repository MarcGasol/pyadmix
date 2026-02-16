# tests/test_cli.py
from click.testing import CliRunner
from pyadmix.cli import main

def test_help_menu():
    """Test that the CLI help menu generates correctly."""
    runner = CliRunner()
    # Simulate the user typing: pyadmix --help
    result = runner.invoke(main, ['--help'])
    
    # Assert the command succeeded (exit code 0 means no errors)
    assert result.exit_code == 0
    # Assert that our tool's description is in the output
    assert "PyAdmix: A high-performance CLI" in result.output


def test_dummy_model_execution():
    """Test that parameters are correctly passed to the dummy model."""
    runner = CliRunner()
    # Simulate the user typing: pyadmix --vcf test_data.vcf -k 5 --invariant
    result = runner.invoke(main, ['--vcf', 'test_data.vcf', '-k', '5', '--invariant'])
    
    assert result.exit_code == 0
    
    # Verify the output string contains the exact variables we passed
    assert "Loading genome matrix from : test_data.vcf" in result.output
    assert "Target populations (K)     : 5" in result.output
    assert "Mode: Extracting population-invariant genome features" in result.output