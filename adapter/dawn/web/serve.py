#!/usr/bin/env python3
"""Serves the web viewer (adapter/dawn/out-web) on http://127.0.0.1:8095/ov_dawn.html, uncached.

python3 adapter/dawn/web/serve.py [DIR] [PORT]
"""
import functools
import http.server
import mimetypes
import pathlib
import sys

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parent.parent / "out-web")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8095
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=root)) as httpd:
        print(f"serving {root} on http://127.0.0.1:{port}/ov_dawn.html", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
