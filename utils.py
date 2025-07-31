import pandas as pd
from io import BytesIO
from typing import List, Dict, Any

# these columns must be present in your Excel sheet
REQUIRED_COLS = [
    "Summary",
    "Issue key",
    "Description",
    "Priority",
    "Status",
    "Linked Issues",
]


def load_issues(source: Any) -> pd.DataFrame:
    """
    Reads an Excel file (path or file-like) and returns a cleaned DataFrame.
    Ensures REQUIRED_COLS are present, strips whitespace, and parses Linked Issues.
    """
    df = pd.read_excel(source, engine="openpyxl")

    # validate columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in input: " + ", ".join(missing))

    # cleanup text fields
    df["Issue key"] = df["Issue key"].astype(str).str.strip()
    df["Summary"]   = df["Summary"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()

    # parse comma-separated linked issues into a list
    df["Linked Issues"] = (
        df["Linked Issues"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    )

    return df


def get_issues_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Converts a DataFrame of issues into a list of dicts,
    suitable for semantic-search indexing or passing into generator.
    """
    return df.to_dict(orient="records")


def export_to_excel(
    scored_df: pd.DataFrame,
    testcases_df: pd.DataFrame = pd.DataFrame()
) -> bytes:
    """
    Bundles one or two DataFrames into a single .xlsx in memory.
    Writes 'Scored Issues' and, if provided, 'Test Cases' sheets.
    Returns raw bytes of the workbook.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        scored_df.to_excel(writer, sheet_name="Scored Issues", index=False)
        if not testcases_df.empty:
            testcases_df.to_excel(writer, sheet_name="Test Cases", index=False)
    return buffer.getvalue()


# -- Optional helpers for persistence & blobs -- #

def save_faiss_index(index, path: str) -> None:
    """
    Persists a FAISS index to disk for later reuse.
    """
    import faiss
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    """
    Loads a previously saved FAISS index from disk.
    """
    import faiss
    return faiss.read_index(path)


def read_blob(file_path: str) -> bytes:
    """
    Reads any file (e.g. screenshot, attachment) as raw bytes (a blob).
    Useful if you later want to attach or version-control assets per issue.
    """
    with open(file_path, "rb") as f:
        return f.read()


def store_blob(data: bytes, dest_path: str) -> None:
    """
    Writes raw bytes to disk.  
    You can use this to persist images, logs or any binary object alongside your test cases.
    """
    with open(dest_path, "wb") as f:
        f.write(data)
