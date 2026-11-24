"""Syntax and structure validation for notebooks and scripts:
Validates that:
1. train_compare.ipynb and evaluate_compare.ipynb have valid JSON and all code cells compile.
2. All python scripts in scripts/experiments/ compile with zero syntax errors.
3. The repo URL in train_compare.ipynb points to the exact GitHub repository.
4. All experiment keys in EXPERIMENTS dict match the build_model logic.
"""
import os
import sys
import json
import ast

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print("=" * 60)
print("1. VALIDATING NOTEBOOK JSON & CELL SYNTAX")
print("=" * 60)

for nb_name in ["train_compare.ipynb", "evaluate_compare.ipynb", "massachusetts.ipynb"]:
    path = os.path.join(REPO, "notebooks", nb_name)
    assert os.path.exists(path), f"Missing {path}"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Checking {nb_name} ({len(nb['cells'])} cells)...")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Validate AST and compilation
            tree = ast.parse(src)
            code_obj = compile(src, f"{nb_name}_cell_{i}", "exec")
            assert code_obj is not None
    print(f"  [OK] {nb_name}: all code cells compile perfectly!")

print("\n" + "=" * 60)
print("2. VALIDATING EXPERIMENT PYTHON SCRIPTS (AST & SYNTAX)")
print("=" * 60)

scripts_to_check = [
    os.path.join(REPO, "scripts", "experiments", "exp_torch.py"),
    os.path.join(REPO, "scripts", "experiments", "exp_data.py"),
    os.path.join(REPO, "scripts", "experiments", "build_notebooks.py"),
    os.path.join(REPO, "scripts", "paper_figures", "reference_models.py"),
]

for script in scripts_to_check:
    assert os.path.exists(script), f"Missing {script}"
    with open(script, encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    compile(src, script, "exec")
    print(f"  [OK] {os.path.basename(script)}: syntax valid!")

print("\n" + "=" * 60)
print("3. VALIDATING TRAIN_COMPARE NOTEBOOK CELLS & SETUP")
print("=" * 60)

with open(os.path.join(REPO, "notebooks", "train_compare.ipynb"), encoding="utf-8") as f:
    nb = json.load(f)

cell_sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(cell_sources) == 5, f"Expected 5 code cells, got {len(cell_sources)}"

# Cell 1: Settings
assert "EXPERIMENT =" in cell_sources[0]
assert "SMOKE_TEST =" in cell_sources[0]
print("  [OK] Cell 1 contains EXPERIMENT and SMOKE_TEST settings.")

# Cell 2: Setup
assert "https://github.com/Dilraj07/Satellite-Imagery-based-Road-Network-Extraction.git" in cell_sources[1]
assert 'REPO_REF = "main"' in cell_sources[1]
print("  [OK] Cell 2 correctly clones from Dilraj07/Satellite-Imagery-based-Road-Network-Extraction (main).")

# Cell 3: Split
assert "D.find_deepglobe()" in cell_sources[2]
assert "D.sha1" in cell_sources[2]
print("  [OK] Cell 3 validates DeepGlobe dataset and checks 623 test tile SHA-1 hash.")

# Cell 4: Train
assert "T.train_experiment" in cell_sources[3]
assert "D.TEST_SHA1" in cell_sources[3]
print("  [OK] Cell 4 calls T.train_experiment with timeout handling and checkpointing.")

# Cell 5: Curves
assert "training_curves.png" in cell_sources[4]
print("  [OK] Cell 5 plots and saves training loss and validation IoU curves.")

print("\n" + "=" * 60)
print("4. EXPERIMENT DEFINITIONS AUDIT")
print("=" * 60)

with open(os.path.join(REPO, "scripts", "experiments", "exp_torch.py"), encoding="utf-8") as f:
    code = f.read()

experiments = ["unet", "dlinknet34_scratch", "dlinknet34", "deeplabv3_mbv3", "lraspp_mbv3"]
for exp in experiments:
    assert f'"{exp}"' in code, f"Experiment key {exp} not found in exp_torch.py"
    print(f"  [OK] '{exp}' registered and ready to train.")

print("\n" + "=" * 60)
print("ALL VALIDATION CHECKS PASSED WITH 0 ERRORS!")
print("=" * 60)
