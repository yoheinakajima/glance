"""Local HTTP server. Binds to 127.0.0.1 and makes no network calls at inference time.

POST /v1/decide  the same Engine.decide() the CLI uses
GET  /v1/models  loaded backends with pinned ids
GET  /healthz    device and load status
"""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, request

from . import __version__
from .backends import model_string
from .config import Config, load_config
from .doctor import run_doctor
from .logging_utils import new_request_id
from .pipeline import Engine
from .prompts import PROMPT_VERSION
from .schema import ErrorResponse


def create_app(cfg: Config | None = None, engine: Engine | None = None, preload: list[str] | None = None) -> Flask:
    cfg = cfg or load_config()
    # Weights come from the local cache at pinned revisions; the server never reaches for the Hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    engine = engine or Engine(cfg, source="server")
    load_errors: dict[str, str] = {}
    for name in preload or []:
        try:
            engine.backend(name)
        except Exception as exc:  # keep serving the backends that did load; /healthz reports the rest
            load_errors[name] = f"{type(exc).__name__}: {exc}"

    app = Flask("glance")
    app.config["engine"] = engine
    doctor = run_doctor(cfg)

    @app.post("/v1/decide")
    def decide() -> Any:
        body = request.get_json(silent=True)
        if body is None:
            payload = ErrorResponse(
                request_id=new_request_id(), code="validation_error", message="request body must be JSON",
                detail=[{"path": "", "message": "body is missing or is not valid JSON"}],
            )
            return jsonify(payload.model_dump()), 422
        status, payload = engine.decide_json(body, source="server")
        return jsonify(payload), status

    @app.get("/v1/models")
    def models() -> Any:
        loaded = [
            {"name": name, "model": model_string(b), "model_id": b.model_id, "revision": b.revision, "kind": b.kind,
             "device": b.device, "dtype": b.dtype, "image_token_budget": b.image_token_budget}
            for name, b in engine.loaded_backends().items()
        ]
        return jsonify({"loaded": loaded, "prompt_version": PROMPT_VERSION})

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({
            "status": "ok" if not load_errors else "degraded",
            "harness_version": __version__,
            "device": doctor["device"], "dtype": doctor["dtype"], "selected_tier": doctor["selected_tier"],
            "loaded": sorted(engine.loaded_backends()), "load_errors": load_errors,
            "prefix_cache": cfg.vlm.prefix_cache,
        })

    return app


def serve(cfg: Config, preload: list[str] | None = None) -> None:
    app = create_app(cfg, preload=preload)
    # threaded=False: one request at a time, which is also what the single-model Engine lock enforces.
    app.run(host=cfg.server.host, port=cfg.server.port, threaded=False, debug=False)
