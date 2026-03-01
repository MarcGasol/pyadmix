# tritonadmix/io/__init__.py
from tritonadmix.io.vcf import load_vcf
from tritonadmix.io.output import write_q_matrix, write_p_matrix

__all__ = ["load_vcf", "write_q_matrix", "write_p_matrix"]
