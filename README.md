# Probabilistic Evaluation for Treatments Plans in Radiation Therapy

This repository is the base module for performing probabilistic evaluation of optimized treatment plans in radiation therapy.
The goal is to provide a clear visualisation and analysis of uncertainties affect the dose distribution and the clinical goals.
This library is not made for optimization purposes, but rather for evaluation of already optimized plans to allow clinicians to make better-informed decisions on
trade-offs between organ-at-risk sparing and target coverage under uncertainty.

**This repository is not intended for clinical use.**

## Installation

```bash
pip install RT-Prob-E
```

### Optional: Streamlit-based UI tools

Two interactive, browser-based tools live in `Probabilistic_Evaluation.ui`
and are available as an optional extra, since they depend on
[Streamlit](https://streamlit.io/) (the results viewer also uses
[streamlit-aggrid](https://pypi.org/project/streamlit-aggrid/) for its
tables):

```bash
pip install RT-Prob-E[ui]
```

**Clinical goals editor** — add, copy, delete and edit the goals in a
clinical goals JSON file:

```python
from Probabilistic_Evaluation.ui import edit_clinical_goals

edit_clinical_goals("clinical_goals.json")  # opens the editor in a browser tab
```

or, from a shell:

```bash
rt-eval-goals clinical_goals.json
```

**Results viewer** — browse a folder of probabilistic evaluation result JSON
files, one browser tab per file, with a draggable, color-coded table of
probabilistic objectives (drag rows to reorder, save/load that order) and a
read-only table of non-probabilistic ones:

```python
from Probabilistic_Evaluation.ui import view_results

view_results("Results")  # opens one tab per *.json file in that folder
```

or, from a shell:

```bash
rt-view-results Results
```

Calling any of these without the `ui` extra installed raises a clear error
telling you to install it, rather than an import traceback.