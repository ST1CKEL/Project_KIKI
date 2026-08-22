"""Best-effort X11 helpers. Silent no-ops on Wayland."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _xid(window: Any) -> int | None:
    try:
        surface = window.get_surface()
    except Exception:
        return None
    if surface is None:
        return None
    getter = getattr(surface, "get_xid", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        return None


def try_move_window(window: Any, x: int, y: int) -> bool:
    xid = _xid(window)
    if xid is None:
        return False
    try:
        import ctypes
        from ctypes import c_char_p, c_int, c_ulong, c_void_p

        xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
        xlib.XOpenDisplay.argtypes = [c_char_p]
        xlib.XOpenDisplay.restype = c_void_p
        xlib.XMoveWindow.argtypes = [c_void_p, c_ulong, c_int, c_int]
        xlib.XFlush.argtypes = [c_void_p]
        dpy = xlib.XOpenDisplay(None)
        if not dpy:
            return False
        try:
            xlib.XMoveWindow(dpy, xid, int(x), int(y))
            xlib.XFlush(dpy)
            return True
        finally:
            xlib.XCloseDisplay(dpy)
    except Exception as exc:
        log.debug("X11 move failed: %s", exc)
        return False


def try_get_position(window: Any) -> tuple[int, int] | None:
    xid = _xid(window)
    if xid is None:
        return None
    try:
        import ctypes
        from ctypes import POINTER, Structure, byref, c_char_p, c_int, c_long, c_ulong, c_void_p

        class Attrs(Structure):
            _fields_ = [
                ("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("height", c_int),
                ("border_width", c_int),
                ("depth", c_int),
                ("visual", c_void_p),
                ("root", c_ulong),
                ("c_class", c_int),
                ("bit_gravity", c_int),
                ("win_gravity", c_int),
                ("backing_store", c_int),
                ("backing_planes", c_ulong),
                ("backing_pixel", c_ulong),
                ("save_under", c_int),
                ("colormap", c_ulong),
                ("map_installed", c_int),
                ("map_state", c_int),
                ("all_event_masks", c_long),
                ("your_event_mask", c_long),
                ("do_not_propagate_mask", c_long),
                ("override_redirect", c_int),
                ("screen", c_void_p),
            ]

        xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
        xlib.XOpenDisplay.argtypes = [c_char_p]
        xlib.XOpenDisplay.restype = c_void_p
        xlib.XGetWindowAttributes.argtypes = [c_void_p, c_ulong, POINTER(Attrs)]
        dpy = xlib.XOpenDisplay(None)
        if not dpy:
            return None
        try:
            attrs = Attrs()
            if xlib.XGetWindowAttributes(dpy, xid, byref(attrs)) == 0:
                return None
            return int(attrs.x), int(attrs.y)
        finally:
            xlib.XCloseDisplay(dpy)
    except Exception as exc:
        log.debug("X11 get position failed: %s", exc)
        return None


def request_keep_above(window: Any, enabled: bool) -> bool:
    """Ask the window manager to pin the window. Returns True if a request was sent."""
    xid = _xid(window)
    if xid is None:
        return False
    try:
        return _send_net_wm_state(xid, enabled)
    except Exception as exc:
        log.debug("X11 keep-above failed: %s", exc)
        return False


def _send_net_wm_state(xid: int, enabled: bool) -> bool:
    import ctypes
    from ctypes import Structure, byref, c_char_p, c_int, c_long, c_ulong, c_void_p

    class XClientMessageEvent(Structure):
        _fields_ = [
            ("type", c_int),
            ("serial", c_ulong),
            ("send_event", c_int),
            ("display", c_void_p),
            ("window", c_ulong),
            ("message_type", c_ulong),
            ("format", c_int),
            ("data", c_long * 5),
        ]

    xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
    xlib.XOpenDisplay.argtypes = [c_char_p]
    xlib.XOpenDisplay.restype = c_void_p
    xlib.XDefaultRootWindow.argtypes = [c_void_p]
    xlib.XDefaultRootWindow.restype = c_ulong
    xlib.XInternAtom.argtypes = [c_void_p, c_char_p, c_int]
    xlib.XInternAtom.restype = c_ulong
    xlib.XSendEvent.argtypes = [c_void_p, c_ulong, c_int, c_long, c_void_p]
    xlib.XSendEvent.restype = c_int
    xlib.XFlush.argtypes = [c_void_p]

    dpy = xlib.XOpenDisplay(None)
    if not dpy:
        return False
    try:
        root = xlib.XDefaultRootWindow(dpy)
        msg = xlib.XInternAtom(dpy, b"_NET_WM_STATE", 0)
        atom = xlib.XInternAtom(dpy, b"_NET_WM_STATE_ABOVE", 0)
        ev = XClientMessageEvent()
        ev.type = 33  # ClientMessage
        ev.window = xid
        ev.message_type = msg
        ev.format = 32
        ev.data[0] = 1 if enabled else 0
        ev.data[1] = atom
        ev.data[2] = 0
        ev.data[3] = 1
        ev.data[4] = 0
        mask = (1 << 20) | (1 << 19)
        xlib.XSendEvent(dpy, root, 0, mask, ctypes.cast(byref(ev), c_void_p))
        xlib.XFlush(dpy)
        return True
    finally:
        xlib.XCloseDisplay(dpy)
