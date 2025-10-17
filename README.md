[README.md](https://github.com/user-attachments/files/22962156/README.md)
# Adversarial Impact on Droop‑Based DQN Controller

## Publication (Resilience Week 2025)

**Impact Analysis of Adversarial Attacks on Deep Reinforcement Learning-Based Wide-Area Microgrid Control**  
**Authors:** Zain ul Abdeen, Suman Rath, Vivek Kumar Singh  
**Affiliation:** National Renewable Energy Laboratory, Golden, Colorado  
**Status:** Accepted at **Resilience Week 2025**  

> This repository accompanies the paper above. The curated notebook reflects the experimental workflow used to analyze the impact of adversarial attacks on a droop-based, DQN-controlled wide-area microgrid.

This repository contains an **impact analysis of adversarial attacks** on a **droop-based  DRL controller** implemented with a **Deep Q-Network**.  
All code is provided in a single Jupyter notebook for clarity and easy reproduction.

## What’s inside
- **Notebook** implementing experiments and analysis of adversarial attacks on a droop‑based DQN controller.
- **Cleaned notebook** with cell outputs removed (recommended for commits).
- **Requirements** to reproduce the environment.
- Optional **HTML export** for quick viewing without Jupyter (see `docs/`).

## Quick start

### 1) Clone
```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

### 2) Create environment
Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name droop-dqn-adversarial
```

(Or with Conda:)
```bash
conda create -n droop-dqn-adversarial python=3.10 -y
conda activate droop-dqn-adversarial
pip install -r requirements.txt
python -m ipykernel install --user --name droop-dqn-adversarial
```

### 3) Run
```bash
jupyter notebook notebooks/Adversarial_impact_on_Single_DQN_final_clean.ipynb
```

> **Tip:** If you prefer committing notebooks without large embedded outputs, keep using the `_clean.ipynb`.  
> You can regenerate a clean copy anytime:
> ```bash
> jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/Adversarial_impact_on_Single_DQN_final.ipynb
> ```

## Reproducing experiments
- Ensure any dataset or simulator dependencies are available (e.g., power systems tools such as `pandapower` or `OpenDSSDirect.py` if used).
- Configure paths and seeds at the top of the notebook if applicable.
- Run cells top‑to‑bottom.

## Repository layout
```
.
├─ notebooks/
│  ├─ Adversarial_impact_on_Single_DQN_final_clean.ipynb  # outputs stripped
│  └─ Adversarial_impact_on_Single_DQN_final.ipynb        # original notebook
├─ docs/
│  └─ Adversarial_impact_on_Single_DQN_final.html         # optional HTML view
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## License
Choose a license (e.g., MIT, Apache‑2.0) and add a `LICENSE` file.  
If unsure, MIT is a simple permissive choice.

---

*Generated on 2025-10-17*


## Curated Notebook (outputs preserved)
- `notebooks/Adversarial_impact_on_Single_DQN_final_curated.ipynb` — redundant code removed (imports consolidated, earlier duplicate function defs removed, later seeds dropped), and descriptive markdown added. Results/outputs remain intact.
