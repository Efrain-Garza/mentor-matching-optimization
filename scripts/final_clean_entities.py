from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE = Path("data/cleaned")

STUDENT_PATH = BASE / "student_clean_ids.parquet"
MENTOR_PATH  = BASE / "mentor_clean_ids.parquet"


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace(
                    to_replace=r"^(nan|none|null)?$",
                    value=pd.NA,
                    regex=True,
                )
            )
    return df


def main():
    # ------------------
    # Students
    # ------------------
    students = pd.read_parquet(STUDENT_PATH)
    students = normalize_strings(students)

    students.to_parquet(STUDENT_PATH, index=False)

    # ------------------
    # Mentors
    # ------------------
    mentors = pd.read_parquet(MENTOR_PATH)
    mentors = normalize_strings(mentors)

    mentors.to_parquet(MENTOR_PATH, index=False)

    # ------------------
    # Sanity checks
    # ------------------
    print("Students shape:", students.shape)
    print("Mentors shape :", mentors.shape)

    print("\nStudent null summary:")
    print(students.isna().sum())

    print("\nMentor null summary:")
    print(mentors.isna().sum())

    print("\nSample students:")
    print(students.head(3).to_string(index=False))

    print("\nSample mentors:")
    print(mentors.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
