import pandas as pd
from functools import lru_cache
from typing import Dict, List

# score weight according to req. sheet
PRIORITY_MAP = {
    "Critical": 10,
    "High":     8,
    "Major":    6,
    "Minor":    4,
    "Low":      2,
}
STATUS_MAP = {
    "To Do":       1,
    "In Progress": 2,
    "Done":        0,
}

def calculate_scores(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate a risk score for each issue.  
    Base score = priority + status, plus half of each linked-issue's score.
    Cycles are safely handled (no infinite loops).
    """
    # Build an in-memory lookup
    lookup = {
        row["Issue key"]: (
            PRIORITY_MAP.get(row["Priority"], 0)
          + STATUS_MAP.get(row["Status"],    0),
            row["Linked Issues"]  # already a list from utils.load_issues
        )
        for _, row in df.iterrows()
    }

    @lru_cache(maxsize=None)
    def _score(key: str, path: tuple = ()) -> float:
        # missing keys or cycles contribute 0
        if key not in lookup or key in path:
            return 0.0

        base, links = lookup[key]
        # recursively add half-weight from each linked issue
        linked_sum = sum(0.5 * _score(link, path + (key,)) for link in links)
        return base + linked_sum

    # compute for all keys
    return {key: _score(key) for key in lookup}
