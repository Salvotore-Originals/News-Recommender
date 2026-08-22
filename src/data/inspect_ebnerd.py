from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EBNERD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_small"
)


FILES = {
    "articles": EBNERD_ROOT / "articles.parquet",
    "train_behaviors": EBNERD_ROOT / "train" / "behaviors.parquet",
    "train_history": EBNERD_ROOT / "train" / "history.parquet",
    "validation_behaviors": (
        EBNERD_ROOT / "validation" / "behaviors.parquet"
    ),
    "validation_history": (
        EBNERD_ROOT / "validation" / "history.parquet"
    ),
}


def inspect_file(name: str, path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)
    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        print("ERROR: File does not exist.")
        return

    df = pd.read_parquet(path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())

    print("\n" + "-" * 80)


def main() -> None:
    print("EB-NeRD SMALL — DATASET INSPECTION")
    print(f"Dataset root: {EBNERD_ROOT}")

    for name, path in FILES.items():
        inspect_file(name, path)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()