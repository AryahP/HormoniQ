"""Local HormoniQ screening application.

Uses the included PCOS Dataset.csv to train a small logistic-regression model at
startup. It is intentionally a screening aid, not a medical diagnosis.
"""
from __future__ import annotations

import csv
import json
import math
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "PCOS Dataset.csv"
EXCLUDED = {
    "PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)",
    "Blood Group", "II    beta-HCG(mIU/mL)", "TSH (mIU/L)", "Waist:Hip Ratio",
}


def sigmoid(value: float) -> float:
    value = max(-35, min(35, value))
    return 1 / (1 + math.exp(-value))


class ScreeningModel:
    def __init__(self) -> None:
        self.features: list[str] = []
        self.medians: dict[str, float] = {}
        self.means: list[float] = []
        self.scales: list[float] = []
        self.weights: list[float] = []
        self.bias = 0.0
        self.train()

    def train(self) -> None:
        with DATASET.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.reader(source))
        headers = [value.strip() for value in rows[0]]
        # Keep duplicate column names addressable, like pandas does.
        seen: dict[str, int] = {}
        for index, name in enumerate(headers):
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                headers[index] = f"{name}.{seen[name] - 1}"
        label_index = headers.index("PCOS (Y/N)")
        self.features = [name for name in headers if name not in EXCLUDED]
        feature_indices = [headers.index(name) for name in self.features]

        raw: list[list[float | None]] = []
        labels: list[float] = []
        for row in rows[1:]:
            if len(row) <= label_index:
                continue
            try:
                label = float(row[label_index])
            except ValueError:
                continue
            values: list[float | None] = []
            for index in feature_indices:
                try:
                    values.append(float(row[index].strip()))
                except (ValueError, IndexError):
                    values.append(None)
            raw.append(values)
            labels.append(label)

        for i, name in enumerate(self.features):
            available = [row[i] for row in raw if row[i] is not None]
            self.medians[name] = float(median(available))
        values = [[self.medians[self.features[i]] if value is None else value
                   for i, value in enumerate(row)] for row in raw]
        count = len(values)
        self.means = [sum(row[i] for row in values) / count for i in range(len(self.features))]
        self.scales = [max(1e-6, math.sqrt(sum((row[i] - self.means[i]) ** 2 for row in values) / count))
                       for i in range(len(self.features))]
        x = [[(row[i] - self.means[i]) / self.scales[i] for i in range(len(self.features))] for row in values]

        # Deterministic, lightly regularised batch logistic regression.
        self.weights = [0.0] * len(self.features)
        self.bias = math.log((sum(labels) + 1) / (count - sum(labels) + 1))
        rate = 0.08
        for _ in range(900):
            gradients = [0.0] * len(self.features)
            bias_gradient = 0.0
            for row, target in zip(x, labels):
                error = sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, row))) - target
                bias_gradient += error
                for i, value in enumerate(row):
                    gradients[i] += error * value
            self.bias -= rate * bias_gradient / count
            for i in range(len(self.weights)):
                self.weights[i] -= rate * (gradients[i] / count + 0.015 * self.weights[i])

    def predict(self, supplied: dict[str, object]) -> float:
        values = []
        for name in self.features:
            value = supplied.get(name, self.medians[name])
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = self.medians[name]
            values.append((value - self.means[len(values)]) / self.scales[len(values)])
        return sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, values)))


MODEL = ScreeningModel()


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/screen":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            probability = MODEL.predict(payload if isinstance(payload, dict) else {})
            body = json.dumps({"probability_pcos": round(probability, 4)}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Please check the values and try again.")


if __name__ == "__main__":
    print("HormoniQ is ready at http://localhost:8080")
    ThreadingHTTPServer(("localhost", 8080), AppHandler).serve_forever()
