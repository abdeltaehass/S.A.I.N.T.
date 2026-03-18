"""
Flask REST API — low-latency inference endpoint with Redis caching.

Endpoints:
  POST /predict          — single connection classification
  POST /predict/batch    — batch of connections (up to 512)
  GET  /stats            — agent memory stats
  GET  /decisions        — recent decisions log (last N)
  POST /review/<id>      — human-in-the-loop review submission
  GET  /health           — liveness probe
"""

import hashlib
import json
import time
from pathlib import Path

import redis
from flask import Flask, jsonify, request
from flask_cors import CORS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FLASK_DEBUG, FLASK_HOST, FLASK_PORT,
    INFERENCE_CACHE_TTL, REDIS_DB, REDIS_HOST, REDIS_PORT,
)
from data.loader import load_artifacts, preprocess_single
from model.classifier import load_model
from agent.reasoning import ThreatReasoningAgent

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# --- Redis ---
_redis: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    return _redis

# --- Model + Agent (lazy-loaded once) ---
_agent: ThreatReasoningAgent | None = None
_feature_cols: list[str] | None = None
_scaler = None
_encoders = None

def get_agent() -> tuple[ThreatReasoningAgent, list[str]]:
    global _agent, _feature_cols, _scaler, _encoders
    if _agent is None:
        print("Loading model and artifacts...")
        model = load_model()
        _scaler, _encoders = load_artifacts()
        feat_path = Path("model/feature_cols.json")
        if not feat_path.exists():
            raise RuntimeError(
                "model/feature_cols.json not found. Run scripts/train.py first."
            )
        with open(feat_path) as f:
            _feature_cols = json.load(f)
        _agent = ThreatReasoningAgent(model)
        print("Agent ready.")
    return _agent, _feature_cols


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(features: dict) -> str:
    return "saint:pred:" + hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()


def _cached_predict(features: dict, agent: ThreatReasoningAgent, feature_cols: list[str]) -> dict:
    r = get_redis()
    key = _cache_key(features)
    try:
        cached = r.get(key)
        if cached:
            result = json.loads(cached)
            result["cached"] = True
            return result
    except redis.RedisError:
        pass  # Redis unavailable — fall through to fresh inference

    X = preprocess_single(features, _scaler, _encoders, feature_cols)
    decision = agent.decide(X)
    result = decision.to_dict()
    result["cached"] = False

    serialized = json.dumps(result)
    try:
        r.setex(key, INFERENCE_CACHE_TTL, serialized)
        r.lpush("saint:decisions", serialized)
        r.ltrim("saint:decisions", 0, 999)
    except redis.RedisError:
        pass

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    redis_ok = False
    try:
        get_redis().ping()
        redis_ok = True
    except Exception:
        pass
    return jsonify({"status": "ok", "redis": redis_ok, "ts": time.time()})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty body"}), 400

    t0 = time.perf_counter()
    agent, feature_cols = get_agent()
    result = _cached_predict(data, agent, feature_cols)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400
    if len(data) > 512:
        return jsonify({"error": "Batch size exceeds 512"}), 400

    t0 = time.perf_counter()
    agent, feature_cols = get_agent()
    results = [_cached_predict(conn, agent, feature_cols) for conn in data]
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify({"results": results, "total_latency_ms": elapsed, "count": len(results)})


@app.route("/stats")
def stats():
    agent, _ = get_agent()
    return jsonify(agent.stats())


@app.route("/decisions")
def decisions():
    n = min(int(request.args.get("n", 100)), 500)
    agent, _ = get_agent()
    return jsonify(agent.recent_decisions(n))


@app.route("/review/<decision_id>", methods=["POST"])
def submit_review(decision_id: str):
    """Human-in-the-loop: analyst submits ground-truth label for a flagged decision."""
    body = request.get_json(force=True) or {}
    analyst_label = body.get("label")
    notes = body.get("notes", "")
    if not analyst_label:
        return jsonify({"error": "Missing 'label' field"}), 400

    r = get_redis()
    review = {
        "decision_id": decision_id,
        "analyst_label": analyst_label,
        "notes": notes,
        "reviewed_at": time.time(),
    }
    try:
        r.hset("saint:reviews", decision_id, json.dumps(review))
    except redis.RedisError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "review_saved", **review})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
