"""Optional interactive tooling, installed with the ``ui`` extra (``pip install rt-probabilistic-evaluation[ui]``).

Importing this subpackage never requires the extra's dependencies to be
installed; only actually launching a tool does (each entry point checks for
``streamlit`` and raises a clear error if it's missing).
"""

from . import clinicalGoalsEditor
from . import resultsViewer
from .clinicalGoalsEditor import edit_clinical_goals
from .resultsViewer import view_results

__all__ = ["clinicalGoalsEditor", "resultsViewer", "edit_clinical_goals", "view_results"]
