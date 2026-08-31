"""Push-to-talk / global shortcut: Super+K → kiki-audio in a few milliseconds."""

from __future__ import annotations

import socket
import sys

from kiki.ipc.paths import socket_path


def trigger() -> int:
    path = socket_path("audio")
    if not path.exists():
        print(f"kiki-audio hört nicht auf {path}", file=sys.stderr)
        return 1
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.connect(str(path))
        sock.sendall(b'{"command":"trigger_turn","source":"hotkey"}\n')
    except Exception as exc:
        print(f"Trigger fehlgeschlagen: {exc}", file=sys.stderr)
        return 1
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(trigger())
