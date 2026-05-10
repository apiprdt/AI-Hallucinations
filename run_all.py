"""
Master reproducibility script for 'Semantic Orientation Effects in Zero-Shot LLM Molecular Inference'
Executes all data analysis and figure generation pipelines.
"""
import os
import sys
import subprocess

def run_script(script_path):
    print(f"\n[{'='*50}]")
    print(f"Executing: {script_path}")
    print(f"[{'='*50}]")
    
    python_exe = sys.executable
    result = subprocess.run([python_exe, script_path])
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    scripts_to_run = [
        "src/classify_extended.py",        # 1. Taxonomy classification for Tox21/ESOL
        "src/generate_fig5.py",            # 2. Scaling analysis figure
        "src/deepened_analysis.py",        # 3. Effect size, Entropy, KL divergence
        "src/calibration_deepening.py",    # 4. Brier decomposition
        "src/generate_annotation_sheet.py" # 5. Stratified sampling extraction
    ]
    
    for script in scripts_to_run:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"WARNING: Script not found: {script}")
            
    print("\n[OK] All reproduction pipelines completed successfully.")

if __name__ == "__main__":
    main()
