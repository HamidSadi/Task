"""Serve the Iris Insights model predictions over a lightweight HTTP API."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from iris_insights import (  # noqa: E402  - imported after sys.path manipulation
    FEATURE_NAMES,
    SPECIES_NAMES,
    FeaturePipeline,
    IrisClassifier,
    load_dataset,
    split_features_target,
)


def _prepare_model() -> tuple[FeaturePipeline, Any]:
    dataset = load_dataset()
    feature_rows, target = split_features_target(dataset)
    pipeline = FeaturePipeline.default(FEATURE_NAMES)
    transformed = pipeline.fit_transform(feature_rows)
    classifier = IrisClassifier(
        pipeline,
        algorithm="random_forest",
        n_estimators=30,
        max_depth=4,
        seed=123,
    )
    result = classifier.train(transformed, target)
    return pipeline, result.best_estimator


PIPELINE, MODEL = _prepare_model()


def _normalise_features(payload: Any) -> List[Dict[str, float]]:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, Iterable):
        raise ValueError("`features` must be a mapping or list of mappings.")
    records: List[Dict[str, float]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Item {index} is not a feature mapping.")
        record: Dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name not in row:
                raise ValueError(f"Feature '{name}' missing from item {index}.")
            record[name] = float(row[name])
        records.append(record)
    return records


class IrisAPIHandler(BaseHTTPRequestHandler):
    server_version = "IrisInsightsHTTP/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        return  # silence default stdout logging

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - inherited name
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - inherited name
        if self.path != "/predict":
            self.send_error(404, "Not found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length header")
            return
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Request body must be valid JSON")
            return
        if not isinstance(payload, dict):
            self.send_error(400, "Request body must be a JSON object")
            return
        if "features" not in payload:
            self.send_error(400, "Missing 'features' field")
            return
        try:
            feature_rows = _normalise_features(payload["features"])
        except ValueError as exc:
            self.send_error(400, str(exc))
            return
        transformed = PIPELINE.transform(feature_rows)
        predictions = MODEL.predict(transformed)
        response = {
            "predictions": predictions,
            "labels": [SPECIES_NAMES[label] for label in predictions],
        }
        self._send_json(response)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), IrisAPIHandler)
    print(f"Serving Iris Insights predictions on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()

