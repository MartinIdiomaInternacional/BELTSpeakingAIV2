import csv, os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")

def log_result(timestamp, task_id, seconds, level, explanation, recommendations, source="api"):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp_utc","task_id","seconds","level","explanation","recommendations","source"])
        w.writerow([timestamp,task_id,f"{seconds:.2f}",level,explanation,recommendations,source])
