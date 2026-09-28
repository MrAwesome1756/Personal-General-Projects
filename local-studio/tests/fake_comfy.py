"""A tiny stand-in for the ComfyUI HTTP API, used by the tests (no GPU needed)."""

import base64
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
KNOWN_NODES = ["KSampler", "CheckpointLoaderSimple", "EmptyLatentImage", "CLIPTextEncode", "VAEDecode",
               "SaveImage", "LoadImage", "LoraLoaderModelOnly", "UNETLoader"]


class State:
    def __init__(self):
        self.prompts = {}
        self.uploads = []
        self.freed = 0


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/system_stats":
                return self._json({"devices": [{"name": "fake-gpu", "vram_total": 1, "vram_free": 1}]})
            if url.path == "/object_info":
                return self._json({n: {} for n in KNOWN_NODES})
            if url.path.startswith("/history/"):
                pid = url.path.split("/")[-1]
                wf = state.prompts.get(pid)
                if wf is None:
                    return self._json({})
                outputs = {}
                for nid, node in wf.items():
                    if node["class_type"] == "SaveImage":
                        prefix = node["inputs"]["filename_prefix"].replace("/", "_")
                        outputs[nid] = {"images": [{"filename": f"{prefix}_00001_.png", "subfolder": "", "type": "output"}]}
                return self._json({pid: {"status": {"status_str": "success", "completed": True}, "outputs": outputs}})
            if url.path == "/view":
                q = parse_qs(url.query)
                assert q.get("filename")
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(PNG)))
                self.end_headers()
                return self.wfile.write(PNG)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if self.path == "/prompt":
                wf = json.loads(body)["prompt"]
                missing = [n["class_type"] for n in wf.values() if n["class_type"] not in KNOWN_NODES]
                if missing:
                    return self._json({"error": "invalid", "node_errors": {"missing": missing}}, 400)
                pid = str(uuid.uuid4())
                state.prompts[pid] = wf
                return self._json({"prompt_id": pid, "number": len(state.prompts), "node_errors": {}})
            if self.path == "/upload/image":
                assert b'name="image"' in body
                state.uploads.append(body)
                return self._json({"name": "keyframe.png", "subfolder": "", "type": "input"})
            if self.path == "/free":
                state.freed += 1
                return self._json({})
            return self._json({"error": "not found"}, 404)

    return Handler


def start():
    state = State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state, f"http://127.0.0.1:{server.server_port}"
