from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

IN_PATH = Path("data/raw/Mentor_Student.xlsx")
OUT_STUD = Path("data/cleaned/student_clean.parquet")
OUT_MENT = Path("data/cleaned/mentor_clean.parquet")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_id(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", "", s.strip())

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH.resolve()}")

    student = pd.read_excel(IN_PATH, sheet_name="Student")
    mentor = pd.read_excel(IN_PATH, sheet_name="Mentor")

    student = clean_columns(student)
    mentor = clean_columns(mentor)

    # --- Student canonicalization ---
    student = student.rename(columns={
        "Standarized_Major": "standardized_major",
        "SM_referen": "standardized_major_id",
        "Additional Info to Consider": "additional_info_to_consider",
        "Additional Info to Consider ": "additional_info_to_consider",
    })

    if "standardized_major_id" in student.columns:
        student["standardized_major_id"] = student["standardized_major_id"].astype(str).map(norm_id)

# Drop unnamed Excel artifacts
    student = student.loc[:, ~student.columns.str.startswith("Unnamed")]
    mentor = mentor.loc[:, ~mentor.columns.str.startswith("Unnamed")]

    # --- Mentor canonicalization ---
    mentor = mentor.rename(columns={
        "Standardrized": "standardized_degree",
    })

    degree_cols = ["1st Degree", "2nd Degree", "3rd Degree", "4th Degree"]
    present = [c for c in degree_cols if c in mentor.columns]
    if present:
        mentor["degree_count"] = mentor[present].notna().sum(axis=1)

    OUT_STUD.parent.mkdir(parents=True, exist_ok=True)
    student.to_parquet(OUT_STUD, index=False)
    mentor.to_parquet(OUT_MENT, index=False)

    print("Wrote:")
    print(" ", OUT_STUD.resolve())
    print(" ", OUT_MENT.resolve())
    print("\nStudent shape:", student.shape)
    print("Mentor shape:", mentor.shape)
    print("\nStudent columns:", student.columns.tolist())
    print("Mentor columns:", mentor.columns.tolist())

if __name__ == "__main__":
    main()
