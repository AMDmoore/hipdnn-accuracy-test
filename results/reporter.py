"""Result collection and reporting for the accuracy test framework."""

import csv
import json
import os
from datetime import datetime


class ResultCollector:
    """Collects test results and writes CSV summary + JSON detail + log files."""

    def __init__(self, output_dir: str, model_name: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self._csv_rows = []
        self._detail_records = []

    def record(
        self,
        test_name: str,
        ep: str,
        seq_len: int,
        config_file: str,
        metrics: dict,
        stdout: str,
        stderr: str,
        success: bool,
        error_msg: str = "",
    ):
        """Record a single test run result."""
        timestamp = datetime.now().isoformat()

        log_filename = (
            f"{test_name}_{ep}_seq{seq_len}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_path = os.path.join(self.log_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== {test_name} | ep={ep} | seq_len={seq_len} "
                    f"| {timestamp} ===\n")
            f.write(f"config_file: {config_file}\n")
            f.write(f"success: {success}\n")
            if error_msg:
                f.write(f"error: {error_msg}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(stdout)
            if stderr:
                f.write("\n=== STDERR ===\n")
                f.write(stderr)

        if success:
            for metric_name, value in metrics.items():
                self._csv_rows.append({
                    "model_name": self.model_name,
                    "ep": ep,
                    "seq_len": seq_len,
                    "config_file": config_file,
                    "test": test_name,
                    "metric": metric_name,
                    "value": value,
                    "timestamp": timestamp,
                })

        self._detail_records.append({
            "model_name": self.model_name,
            "test": test_name,
            "ep": ep,
            "seq_len": seq_len,
            "config_file": config_file,
            "success": success,
            "error": error_msg,
            "metrics": metrics,
            "log_file": log_filename,
            "timestamp": timestamp,
        })

    def write_summary(self):
        """Write CSV summary and JSON detail files."""
        csv_path = os.path.join(self.output_dir, "results_summary.csv")
        fieldnames = [
            "model_name", "ep", "seq_len", "config_file",
            "test", "metric", "value", "timestamp",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._csv_rows)

        json_path = os.path.join(self.output_dir, "results_detail.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._detail_records, f, indent=2, ensure_ascii=False)

        print(f"\nResults written to: {self.output_dir}")
        print(f"  CSV summary : {csv_path}")
        print(f"  JSON detail : {json_path}")
        print(f"  Logs        : {self.log_dir}")
