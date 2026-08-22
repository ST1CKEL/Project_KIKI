"""KIKI's own LLM harness — neither Ollama nor llama.cpp.

Runs as a separate service for the same reason the TTS service does: PyTorch and
CUDA never enter the GTK process, so a CUDA fault takes down one unit that
systemd restarts, not the whole desktop pet.

What owning the harness buys over a third-party server:

* **Token-level control.** Reasoning is suppressed by banning the `<think>`
  token outright, not by prefilling text and hoping the template cooperates.
* **Slots.** Concurrent requests are admitted by priority, so a background
  summary cannot stall a conversation.
* **A VRAM budget.** The harness knows what it holds and what it may evict,
  sharing the card with the TTS service instead of fighting it.

Loopback only, like every other KIKI service.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batching import DONE, BatchScheduler, Sequence  # noqa: E402
from toolcalls import ToolCallStreamParser  # noqa: E402
from vram import Priority, Resident, VramBudget  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18770
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_NEW_TOKENS = 2048

# Emitted by Qwen3 to open a reasoning block. Banning the id is the honest
# version of prefilling "<think></think>": the model cannot start one at all.
THINK_OPENERS = ("<think>", "<thinking>")


class HarnessError(Exception):
    """Generation failed or the model is not loaded."""


class EchoEngine:
    """Stand-in for smoke tests and CI: no torch, no GPU, no download."""

    model_id = "echo"
    ready = True
    loaded_bytes = 0

    def generate(self, messages: list[dict], *, tools: list | None = None, **_kw):
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        already_ran = any(m.get("role") == "tool" for m in messages)
        if tools and not already_ran and "werkzeug" in str(last.get("content", "")).lower():
            # One deterministic call, then an answer — so a full agent loop can
            # be exercised without a GPU instead of spinning to the step limit.
            name = str((tools[0].get("function") or {}).get("name") or "unknown")
            yield '<tool_call>{"name": "' + name + '", "arguments": {}}</tool_call>'
            return
        if already_ran:
            result = next(m for m in reversed(messages) if m.get("role") == "tool")
            yield f"Ergebnis: {result.get('content', '')}"
            return
        for piece in f"Echo: {last.get('content', '')}".split(" "):
            yield piece + " " 

    def unload(self) -> None:
        return None


class TransformersEngine:
    """Qwen3 on PyTorch, driven directly."""

    def __init__(self, model_id: str, *, device: str = "auto", quantize: str = "none") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.ready = False
        resolved = self._pick_device(device)
        log.info("loading %s on %s (quantize=%s)", model_id, resolved, quantize)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": resolved}
        if quantize in {"int4", "int8"}:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantize == "int4", load_in_8bit=quantize == "int8"
                )
                kwargs.pop("dtype", None)
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise HarnessError(
                    "bitsandbytes fehlt für Quantisierung. "
                    "Im venv nachinstallieren oder --quantize none verwenden."
                ) from exc

        before = self._free()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.loaded_bytes = max(0, before - self._free())
        self._banned = self._banned_token_ids()
        self.ready = True
        log.info(
            "ready model=%s vram=%.2f GB banned_reasoning_tokens=%s",
            model_id,
            self.loaded_bytes / 1e9,
            len(self._banned),
        )

    def _pick_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda:0" if self._torch.cuda.is_available() else "cpu"

    def _free(self) -> int:
        if not self._torch.cuda.is_available():
            return 0
        free, _total = self._torch.cuda.mem_get_info()
        return int(free)

    def _banned_token_ids(self) -> list[list[int]]:
        """Token ids that would open a reasoning block."""
        banned: list[list[int]] = []
        for marker in THINK_OPENERS:
            ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if ids:
                banned.append(ids)
        return banned

    def unload(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        del self.model
        self.ready = False
        self._torch.cuda.empty_cache()
        log.info("unloaded %s", self.model_id)

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        suppress_reasoning: bool = True,
        tools: list | None = None,
        **_kw,
    ):
        from threading import Thread

        from transformers import TextIteratorStreamer

        if not self.ready:
            raise HarnessError("Modell ist nicht geladen.")
        # Qwen3's own template renders tool declarations and knows the
        # <tool_call> convention. Owning the harness means using it directly
        # instead of describing tools in prose and hoping.
        template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # An older template without tool support: drop them rather than fail.
            log.warning("chat template does not accept tools; continuing without")
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": min(int(max_new_tokens), MAX_NEW_TOKENS),
            "do_sample": temperature > 0,
            "temperature": max(0.01, float(temperature)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if suppress_reasoning and self._banned:
            kwargs["bad_words_ids"] = self._banned

        worker = Thread(target=self._run, args=(kwargs,), daemon=True)
        worker.start()
        yield from streamer

    def _run(self, kwargs: dict[str, Any]) -> None:
        try:
            with self._torch.inference_mode():
                self.model.generate(**kwargs)
        except Exception:  # pragma: no cover - surfaced through the empty stream
            log.exception("generation failed")


class Slots:
    """Admission control. A conversation must not wait behind a summary."""

    def __init__(self, count: int = 2) -> None:
        self._capacity = max(1, int(count))
        self._free = threading.Semaphore(self._capacity)
        self._lock = threading.Lock()
        self._active: dict[str, Priority] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    def active(self) -> dict[str, Priority]:
        with self._lock:
            return dict(self._active)

    def acquire(self, name: str, priority: Priority, *, timeout: float = 120.0) -> bool:
        if not self._free.acquire(timeout=timeout):
            return False
        with self._lock:
            self._active[name] = priority
        return True

    def release(self, name: str) -> None:
        with self._lock:
            existed = self._active.pop(name, None)
        if existed is not None:
            self._free.release()


def _from_scheduler(scheduler, messages, **kwargs):
    """Submit to the batch scheduler and yield its tokens as they arrive."""
    import uuid as _uuid

    sequence = Sequence(id=_uuid.uuid4().hex[:12], messages=messages, **kwargs)
    scheduler.submit(sequence)
    try:
        while True:
            item = sequence.out.get(timeout=600)
            if item is DONE:
                return
            yield item
    except Exception:
        scheduler.cancel(sequence)
        raise


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class LlmHandler(BaseHTTPRequestHandler):
    engine: Any = None
    budget: Any = None
    slots: Any = None
    scheduler: Any = None
    server_version = "kiki-llm/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/", "/health"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        engine = self.engine
        budget = self.budget
        self._send_json(
            200,
            {
                "ok": True,
                "ready": bool(getattr(engine, "ready", False)),
                "model": getattr(engine, "model_id", "unknown"),
                "vram_bytes": int(getattr(engine, "loaded_bytes", 0)),
                "vram_free": int(budget.free_bytes()) if budget else 0,
                "slots": getattr(self.slots, "capacity", 0),
                "busy": {k: v.value for k, v in (self.slots.active() if self.slots else {}).items()},
                "batch": self.scheduler.stats() if self.scheduler else None,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/v1/generate", "/generate"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        engine = self.engine
        if engine is None or not getattr(engine, "ready", False):
            self._send_json(503, {"ok": False, "ready": False, "error": "Modell nicht geladen"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Content-Length fehlt"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"ok": False, "error": "Anfrage zu groß"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "JSON ungültig"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON ist kein Objekt"})
            return

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(400, {"ok": False, "error": "messages fehlen"})
            return
        try:
            priority = Priority(str(payload.get("priority") or "high"))
        except ValueError:
            priority = Priority.HIGH

        name = f"{priority.value}-{time.monotonic_ns()}"
        if self.slots is not None and not self.slots.acquire(name, priority, timeout=1.0):
            self._send_json(
                503, {"ok": False, "error": "Alle Slots belegt", "busy": True}
            )
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            tools = payload.get("tools")
            tools = tools if isinstance(tools, list) else None
            parser = ToolCallStreamParser()
            scheduler = self.scheduler

            def emit(obj: dict) -> None:
                self.wfile.write(_json_bytes(obj) + b"\n")
                self.wfile.flush()

            if scheduler is not None:
                pieces = _from_scheduler(
                    scheduler,
                    messages,
                    tools=tools,
                    temperature=float(payload.get("temperature", 0.7)),
                    max_new_tokens=int(payload.get("max_new_tokens", 512)),
                    suppress_reasoning=bool(payload.get("suppress_reasoning", True)),
                    priority=priority.value,
                )
            else:
                pieces = engine.generate(
                    messages,
                    temperature=float(payload.get("temperature", 0.7)),
                    max_new_tokens=int(payload.get("max_new_tokens", 512)),
                    suppress_reasoning=bool(payload.get("suppress_reasoning", True)),
                    tools=tools,
                )
            for piece in pieces:
                if not piece:
                    continue
                text, calls = parser.feed(piece)
                if text:
                    emit({"delta": text})
                for call in calls:
                    emit({"tool_call": call.as_dict()})
            tail, calls = parser.finish()
            if tail:
                emit({"delta": tail})
            for call in calls:
                emit({"tool_call": call.as_dict()})
            emit({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            log.info("client hung up")
        except Exception:
            log.exception("generation failed")
        finally:
            if self.slots is not None:
                self.slots.release(name)


def _bind_host(host: str) -> str:
    text = host.strip() or DEFAULT_HOST
    if text not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("kiki-llm bindet nur Loopback (127.0.0.1 / ::1).")
    return "127.0.0.1" if text == "localhost" else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIKI local LLM harness")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument("--quantize", default="none", choices=("none", "int8", "int4"))
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="continuous batching with this many sequences per forward pass (0 = off)",
    )
    parser.add_argument("--echo", action="store_true", help="no torch: echo replies")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s kiki-llm: %(message)s",
    )
    host = _bind_host(args.host)

    total = 0
    probe = None
    if not args.echo:
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                probe = lambda: torch.cuda.mem_get_info()[0]  # noqa: E731
        except ImportError:
            log.warning("torch fehlt — VRAM-Budget ohne Messung")
    budget = VramBudget(total_bytes=total or 1, probe=probe)

    if args.echo:
        engine: Any = EchoEngine()
        log.info("echo engine (no torch)")
    else:
        try:
            engine = TransformersEngine(
                args.model, device=args.device, quantize=args.quantize
            )
        except ImportError as exc:
            log.error("torch/transformers fehlen (%s). scripts/setup-llm.sh oder --echo.", exc)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
        budget.register(
            Resident(
                name="llm",
                bytes_used=int(getattr(engine, "loaded_bytes", 0)),
                priority=Priority.HIGH,
                evictable=True,
                unload=engine.unload,
            )
        )

    LlmHandler.engine = engine
    LlmHandler.budget = budget
    LlmHandler.slots = Slots(max(args.slots, args.batch or 0))
    scheduler = None
    if args.batch and not args.echo:
        import torch as _torch
        from torch_batch import TorchBatchedModel

        scheduler = BatchScheduler(
            TorchBatchedModel(
                engine.tokenizer,
                engine.model,
                torch_module=_torch,
                banned_ids=engine._banned,
            ),
            max_batch=int(args.batch),
        )
        scheduler.start()
        log.info("continuous batching on, max_batch=%s", args.batch)
    LlmHandler.scheduler = scheduler
    httpd = ThreadingHTTPServer((host, int(args.port)), LlmHandler)
    log.info(
        "listening on http://%s:%s model=%s slots=%s",
        host,
        args.port,
        getattr(engine, "model_id", "unknown"),
        args.slots,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        if scheduler is not None:
            scheduler.stop()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
