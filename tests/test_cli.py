# tests/test_cli.py

import os
import tempfile
from click.testing import CliRunner
from tritonadmix.cli import main


def test_help_menu():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "TritonAdmix" in result.output


def test_admixture_run():
    runner = CliRunner()
    vcf_path = os.path.join(os.path.dirname(__file__), 'data', 'test.vcf')

    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, 'test_output')
        result = runner.invoke(main, [
            '--vcf', vcf_path,
            '-k', '2',
            '-o', output_prefix,
            '--max-iter', '10',
            '--seed', '42'
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check output files exist
        assert os.path.exists(f"{output_prefix}.2.Q")
        assert os.path.exists(f"{output_prefix}.2.P")

        # Check Q file format (5 individuals, 2 columns)
        with open(f"{output_prefix}.2.Q") as f:
            lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                values = [float(x) for x in line.strip().split()]
                assert len(values) == 2
                assert abs(sum(values) - 1.0) < 0.01  # rows sum to ~1

        # Check P file format (10 SNPs, 2 columns)
        with open(f"{output_prefix}.2.P") as f:
            lines = f.readlines()
            assert len(lines) == 10
            for line in lines:
                values = [float(x) for x in line.strip().split()]
                assert len(values) == 2
                assert all(0 <= v <= 1 for v in values)
