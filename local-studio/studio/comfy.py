"""Minimal ComfyUI HTTP API client (stdlib only).

Endpoints used: /prompt, /history/{id}, /view, /upload/image, /system_stats,
/object_info, /queue, /free. ComfyUI should listen on 127.0.0.1 only.
"""

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url="http://127.0.0.1:8188", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ---- low level -------------------------------------------------------

    def _request(self, method, path, body=None, headers=None, raw=False):
        req = urllib.request.Request(self.base_url + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise ComfyError(f"{method} {path} -> HTTP {e.code}: {detail[:2000]}") from None
        except urllib.error.URLError as e:
            raise ComfyError(f"Cannot reach ComfyUI at {self.base_url} ({e.reason}). Is it running?") from None
        if raw:
            return data
        return json.loads(data) if data else {}

    def _post_json(self, path, payload):
        return self._request("POST", path, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})

    # ---- API -------------------------------------------------------------

    def is_up(self):
        try:
            self.system_stats()
            return True
        except ComfyError:
            return False

    def system_stats(self):
        return self._request("GET", "/system_stats")

    def object_info(self):
        return self._request("GET", "/object_info")

    def queue_state(self):
        return self._request("GET", "/queue")

    def free(self, unload_models=True, free_memory=True):
        """Ask ComfyUI to unload models / free VRAM (so other GPU users get room)."""
        return self._post_json("/free", {"unload_models": unload_models, "free_memory": free_memory})

    def queue_prompt(self, workflow):
        resp = self._post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if resp.get("node_errors"):
            raise ComfyError(f"Workflow rejected: {json.dumps(resp['node_errors'])[:2000]}")
        if "prompt_id" not in resp:
            raise ComfyError(f"Unexpected /prompt response: {resp}")
        return resp["prompt_id"]

    def history(self, prompt_id):
        return self._request("GET", f"/history/{prompt_id}").get(prompt_id)

    def wait(self, prompt_id, timeout_s=900, poll_s=1.5):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            entry = self.history(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    raise ComfyError(f"Execution failed: {json.dumps(msgs)[:2000]}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll_s)
        raise ComfyError(f"Timed out after {timeout_s}s waiting for prompt {prompt_id}")

    def upload_image(self, path, overwrite=True):
        path = Path(path)
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n{'true' if overwrite else 'false'}\r\n".encode(),
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
             f"Content-Type: {ctype}\r\n\r\n").encode() + path.read_bytes() + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        resp = self._request("POST", "/upload/image", b"".join(parts),
                             {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        name = resp.get("name", path.name)
        sub = resp.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    def download(self, file_ref, dest_dir):
        query = urllib.parse.urlencode({
            "filename": file_ref["filename"],
            "subfolder": file_ref.get("subfolder", ""),
            "type": file_ref.get("type", "output"),
        })
        data = self._request("GET", f"/view?{query}", raw=True)
        dest = Path(dest_dir) / file_ref["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    @staticmethod
    def output_files(history_entry):
        """All saved files from a history entry (images, videos, gifs, audio)."""
        files = []
        for node_out in history_entry.get("outputs", {}).values():
            for value in node_out.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "filename" in item and item.get("type", "output") == "output":
                            files.append(item)
        return files

    def run(self, workflow, dest_dir, timeout_s=900):
        """Queue a workflow, wait, download its outputs. Returns (prompt_id, [paths], seconds)."""
        start = time.monotonic()
        prompt_id = self.queue_prompt(workflow)
        entry = self.wait(prompt_id, timeout_s=timeout_s)
        paths = [self.download(f, dest_dir) for f in self.output_files(entry)]
        return prompt_id, paths, round(time.monotonic() - start, 1)
