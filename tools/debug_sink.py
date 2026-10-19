#!/usr/bin/env python3
"""A Godot remote debugger that only listens: accepts the game's --remote-debug connection and discards what it
sends. With a debugger attached, Godot registers its "visual" profiler, which bench.gd --profile=1 turns on: the
renderer then captures a GPU timestamp per pass. Prints the port it listens on, then runs until killed.

    tools/debug_sink.py [port]    (0 or none: any free port)
"""
import socket
import sys
import threading

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 0))
sock.listen(4)
print(sock.getsockname()[1], flush=True)


def drain(conn):
    while conn.recv(1 << 16):
        pass


while True:
    conn, _ = sock.accept()
    threading.Thread(target=drain, args=(conn,), daemon=True).start()
