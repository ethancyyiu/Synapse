"""Serve Synapse locally with browser caching turned off, so edits always show on reload.

Run: python tools/serve.py   then open http://localhost:8766
"""
import functools
import http.server
import pathlib
from urllib.parse import unquote, urlsplit


class NoCache(http.server.SimpleHTTPRequestHandler):
    def private_path(self):
        path = unquote(urlsplit(self.path).path).replace("\\", "/")
        parts = [part for part in path.split("/") if part]
        return (any(part.startswith(".") or ":" in part for part in parts) or
                path.lower().endswith((".py", ".pyc", ".pyo")))

    def do_GET(self):
        if self.private_path():
            return self.send_error(404, "Not found")
        return super().do_GET()

    def do_HEAD(self):
        if self.private_path():
            return self.send_error(404, "Not found")
        return super().do_HEAD()

    def log_message(self, format, *args):
        # Do not record browser-supplied paths or query strings in console logs.
        super().log_message("request handled")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


root = pathlib.Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    http.server.test(HandlerClass=functools.partial(NoCache, directory=str(root)), port=8766,
                     bind="127.0.0.1")
