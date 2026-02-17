"""Generate demo data for testing r4mc report engine."""
import os
import pandas as pd
from pathlib import Path

# Create demo run directory
demo_dir = Path("tests/fixtures/demo_run")
demo_dir.mkdir(parents=True, exist_ok=True)

# Create two arms with episodes
arms = ["reinvent_explore", "mol2mol_exploit"]

for arm in arms:
    arm_dir = demo_dir / arm
    arm_dir.mkdir(exist_ok=True)
    
    # Create 3 episodes per arm
    for ep in range(1, 4):
        ep_dir = arm_dir / f"episode_{ep:03d}"
        ep_dir.mkdir(exist_ok=True)
        
        # Generate synthetic SMILES data
        smiles_list = [
            "CCO",  # ethanol
            "CC(C)O",  # isopropanol
            "c1ccccc1",  # benzene
            "CC(=O)O",  # acetic acid
            "CC(C)(C)O",  # tert-butanol
        ]
        
        scores = [0.8 + ep * 0.05, 0.7, 0.65, 0.6, 0.55]
        
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'Score': scores
        })
        
        df.to_csv(ep_dir / "scaffold_memory.csv", index=False)

# Create a simple log file
log_content = f"""
Episode 1: Selected arm: reinvent_explore
Reasons: ["High UCB score", "Low uncertainty"]
Episode 2: Selected arm: mol2mol_exploit  
Reasons: ["Best reward", "High confidence"]
Episode 3: Selected arm: reinvent_explore
Reasons: ["Exploration needed", "Novelty bonus"]
"""

with open(demo_dir / "controller.log", "w") as f:
    f.write(log_content)

print(f"Demo data generated in {demo_dir}")
