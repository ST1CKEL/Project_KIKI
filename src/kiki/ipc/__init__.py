"""JSON-lines Unix-socket protocol between KIKI voice-first processes."""

from kiki.ipc.client import JsonUnixClient
from kiki.ipc.paths import runtime_dir, socket_path
from kiki.ipc.protocol import dumps, loads

__all__ = ["JsonUnixClient", "dumps", "loads", "runtime_dir", "socket_path"]
