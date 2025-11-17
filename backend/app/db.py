import csv
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")


def log_result(
    timestamp: str,
    task_id: int,
    seconds: float,
    level: str,
    explanation: str,
    recommendations: str,
    source: str = "api",
):
    file_exists = os.path.isfile(RESULTS_FILE)

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
                timestamp,
                task_id,
                f"{seconds:.2f}",
                level,
                explanation,
                recommendations,
                source,
            ]
        )
