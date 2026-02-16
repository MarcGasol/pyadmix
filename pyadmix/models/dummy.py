def run_dummy_model(vcf_path: str, k: int, extract_invariant: bool):
    """
    A dummy function to test CLI parameter passing.
    """
    print("\n" + "="*40)
    print(" ENGINE: Model Execution Started")
    print("="*40)
    
    print(f" [Data] Loading genome matrix from : {vcf_path}")
    print(f" [Math] Target populations (K)     : {k}")
    
    if extract_invariant:
        print(" [Arch] Mode: Extracting population-invariant genome features...")
    else:
        print(" [Arch] Mode: Standard matrix factorization...")
        
    print("="*40)
    print(" ENGINE: Execution Finished\n")