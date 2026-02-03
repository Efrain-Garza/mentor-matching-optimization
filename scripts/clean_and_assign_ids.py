from pathlib import Path
import pandas as pd

BASE = Path("data/cleaned")

STUDENT_IN = BASE / "student_clean.parquet"
MENTOR_IN = BASE / "mentor_clean.parquet"

STUDENT_OUT = BASE / "student_clean_ids.parquet"
MENTOR_OUT = BASE / "mentor_clean_ids.parquet"


def zero_pad(prefix: str, n: int) -> str:
    return f"{prefix}{n:02d}"


def main():
    # -----------------------
    # Students
    # -----------------------
    students = pd.read_parquet(STUDENT_IN).copy()

    students.insert(
        0,
        "student_id",
        [zero_pad("s", i + 1) for i in range(len(students))]
    )

    # Standardize domain IDs: i1_ → i01
    students["standardized_major_id"] = (
        students["standardized_major_id"]
        .astype(str)
        .str.replace("_", "", regex=False)
        .str.replace(r"^i(\d)$", r"i0\1", regex=True)
    )

    students.to_parquet(STUDENT_OUT, index=False)

    # -----------------------
    # Mentors
    # -----------------------
    mentors = pd.read_parquet(MENTOR_IN).copy()

    mentors.insert(
        0,
        "mentor_id",
        [zero_pad("m", i + 1) for i in range(len(mentors))]
    )

    mentors["standardized_degree"] = (
        mentors["standardized_degree"]
        .astype(str)
        .str.replace("_", "", regex=False)
        .str.replace(r"^i(\d)$", r"i0\1", regex=True)
    )

    mentors.to_parquet(MENTOR_OUT, index=False)

    # -----------------------
    # Sanity check
    # -----------------------
    print("Students:", students.shape)
    print("Mentors :", mentors.shape)
    print("\nStudent IDs:", students["student_id"].head().tolist())
    print("Mentor IDs :", mentors["mentor_id"].head().tolist())
    print("Domain IDs :", students["standardized_major_id"].unique()[:5])


if __name__ == "__main__":
    main()
