
import pandas as pd
from functools import lru_cache
from typing import Dict

# score weight according to Req sheet
PRIORITY_MAP = {
    "Critical": 10,
    "High":      8,
    "Major":     6,
    "Minor":     4,
    "Low":       2,
}
STATUS_MAP = {
    "To Do":       1,
    "In Progress": 2,
    "Done":        0,
}


def calculate_scores(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate a risk score for each issue in df.
    Base = PRIORITY_MAP + STATUS_MAP + 0.5 * linked-issues’ scores.
    Cycles are safely handled by lru_cache.
    """
    # build lookup: key -> for linked issue
    lookup = {
        row["Issue key"]: (
            PRIORITY_MAP.get(row["Priority"], 0)
          + STATUS_MAP.get(row["Status"], 0),
            row["Linked Issues"]
        )
        for _, row in df.iterrows()
    }

    @lru_cache(maxsize=None)
    def _score(key: str, path: tuple = ()) -> float:
        if key not in lookup or key in path:
            return 0.0
        base, links = lookup[key]
        linked_sum = sum(0.5 * _score(link, path + (key,)) for link in links)
        return base + linked_sum

    return {k: _score(k) for k in lookup}
