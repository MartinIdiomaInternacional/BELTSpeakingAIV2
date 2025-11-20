import csv
import os
from typing import Optional

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")


def log_result(
    timestamp_utc: str,
    task_id: int,
    seconds: float,
    level: str,
    explanation: str,
    recommendations: str,
    source: str = "api",
) -> None:
    """Append a row to results.csv (created on first use)."""
    file_exists = os.path.isfile(RESULTS_FILE)

    # Ensure directory exists (in case path changes in future)
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "task_id",
                    "seconds",
                    "level",
                    "explanation",
                    "recommendations",
                    "source",
                ]
            )
        writer.writerow(
            [
                timestamp_utc,
                task_id,
                f"{seconds:.2f}",
                level,
                explanation,
                recommendations,
                source,
            ]
        )
