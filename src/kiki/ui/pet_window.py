from __future__ import annotations

import logging
from pathlib import Path

from gi.repository import Gdk, GdkPixbuf, Gio, Graphene, Gtk

from kiki.character.animation_engine import AnimationEngine
from kiki.character.assets import CharacterPack
from kiki.character.state_machine import CharacterState, CharacterStateMachine
from kiki.config.settings import Settings, save_settings
from kiki.platform.capabilities import PlatformCapabilities
from kiki.platform.x11 import request_keep_above, try_get_position, try_move_window
from kiki.ui.input_region import AlphaRegionCache
from kiki.ui.menu_model import build_pet_menu

log = logging.getLogger(__name__)

DRAG_THRESHOLD = 10
HOVER_REACTION_MS = 650


class PetWindow(Gtk.Window):
    def __init__(
        self,
        *,
        application: Gtk.Application,
        pack: CharacterPack,
        machine: CharacterStateMachine,
        settings: Settings,
        capabilities: PlatformCapabilities,
    ) -> None:
        super().__init__(application=application, title="KIKI")
        self._pack = pack
        self._machine = machine
        self._settings = settings
        self._capabilities = capabilities
        self._engine: AnimationEngine = pack.engine()
        self._textures: dict[str, Gdk.Texture] = {}
        self._pixbufs: dict[str, GdkPixbuf.Pixbuf] = {}
        self._input_regions = AlphaRegionCache()
        self._region_size: tuple[int, int] | None = None
        self._moved = False
        self._tick_id = 0
        self._last_frame_time_us: int | None = None
        self._frame_remainder_us = 0

        self.add_css_class("kiki-pet")
        self.remove_css_class("background")
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_deletable(True)
        self.set_focus_on_click(True)
        self.set_default_icon_name("io.github.projectkiki.Kiki")

        self._picture = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN)
        self._picture.add_css_class("kiki-sprite")
        self._picture.set_hexpand(True)
        self._picture.set_vexpand(True)
        self.set_child(self._picture)
        self._apply_scale()

        click = Gtk.GestureClick()
        click.set_button(0)
        click.connect("pressed", self._on_pressed)
        click.connect("released", self._on_released)
        self.add_controller(click)

        drag = Gtk.GestureDrag()
        drag.set_button(1)
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)

        motion = Gtk.EventControllerMotion()
        motion.connect("enter", self._on_pointer_enter)
        motion.connect("leave", self._on_pointer_leave)
        self.add_controller(motion)

        self._unsub = machine.subscribe(self._on_state)
        self.connect("realize", self._on_realize)
        self.connect("close-request", self._on_close)
        self._on_state(machine.state)
        self._tick_id = self.add_tick_callback(self._on_tick)

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._apply_scale()
        self._apply_keep_above()
        self._apply_input_region()

    def reload_pack(self, pack: CharacterPack) -> None:
        self._pack = pack
        self._engine = pack.engine()
        self._textures.clear()
        self._pixbufs.clear()
        self._input_regions.clear()
        self._engine.play(self._machine.state)
        self._show_frame()
        self._apply_scale()

    def _apply_scale(self) -> None:
        height = self._settings.pet_height()
        width = max(80, int(height * self._pack.aspect))
        size = (width, height)
        if size != self._region_size:
            self._input_regions.clear()
            self._region_size = size
        self._picture.set_size_request(width, height)
        self.set_default_size(width, height)

    def _on_state(self, state: CharacterState) -> None:
        self._engine.play(state)
        self._show_frame()

    def _on_tick(self, _widget: Gtk.Widget, frame_clock: Gdk.FrameClock) -> bool:
        now_us = int(frame_clock.get_frame_time())
        previous_us = self._last_frame_time_us
        self._last_frame_time_us = now_us
        if previous_us is None or now_us < previous_us:
            self._frame_remainder_us = 0
            dt_ms = 0
        else:
            elapsed_us = now_us - previous_us + self._frame_remainder_us
            dt_ms, self._frame_remainder_us = divmod(elapsed_us, 1000)
        self._machine.tick()
        if self._machine.paused:
            return True
        if self._engine.advance(dt_ms):
            self._show_frame()
        return True

    def _show_frame(self) -> None:
        path = self._engine.frame.path
        try:
            self._picture.set_paintable(self._texture(path))
        except Exception:
            log.warning("skipping unreadable frame %s", path)
            return
        self._apply_input_region()

    def _texture(self, path: Path) -> Gdk.Texture:
        key = str(path)
        if key not in self._textures:
            try:
                self._textures[key] = Gdk.Texture.new_from_filename(key)
            except Exception:
                log.warning("could not load frame %s", path)
                raise
        return self._textures[key]

    def _pixbuf(self, path: Path) -> GdkPixbuf.Pixbuf:
        key = str(path)
        if key not in self._pixbufs:
            self._pixbufs[key] = GdkPixbuf.Pixbuf.new_from_file(key)
        return self._pixbufs[key]

    def _on_realize(self, *_args: object) -> None:
        self._apply_keep_above()
        self._restore_position()
        self._apply_input_region()

    def _restore_position(self) -> None:
        if not self._capabilities.can_position_window:
            return
        if self._settings.pet.last_x < 0 or self._settings.pet.last_y < 0:
            return
        try_move_window(self, self._settings.pet.last_x, self._settings.pet.last_y)

    def _persist_position(self) -> None:
        pos = try_get_position(self)
        if pos is None:
            return
        self._settings.pet.last_x, self._settings.pet.last_y = pos
        save_settings(self._settings)

    def _apply_keep_above(self) -> None:
        if not self._settings.pet.always_on_top:
            return
        if self._capabilities.can_keep_above:
            request_keep_above(self, True)

    def _apply_input_region(self) -> None:
        surface = self.get_surface()
        if surface is None:
            return
        if not self._settings.pet.click_through_idle:
            surface.set_input_region(None)
            return
        try:
            path = self._engine.frame.path
            region = self._input_regions.get(
                path,
                max(1, self.get_width()),
                max(1, self.get_height()),
                self._pixbuf,
            )
            surface.set_input_region(region)
        except Exception:
            log.debug("input region failed", exc_info=True)

    def _on_pointer_enter(
        self,
        _controller: Gtk.EventControllerMotion,
        _x: float,
        _y: float,
    ) -> None:
        self.set_cursor_from_name("pointer")
        if self._machine.state is CharacterState.IDLE:
            self._machine.set(CharacterState.HAPPY, hold_ms=HOVER_REACTION_MS)

    def _on_pointer_leave(self, _controller: Gtk.EventControllerMotion) -> None:
        self.set_cursor_from_name(None)

    def _on_pressed(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float) -> None:
        button = gesture.get_current_button()
        self._moved = False
        if button == 3:
            self._popup_menu(x, y)
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_released(self, gesture: Gtk.GestureClick, _n_press: int, _x: float, _y: float) -> None:
        if self._moved:
            return
        if gesture.get_current_button() == 1:
            app = self.get_application()
            if app is not None:
                app.activate_action("chat", None)

    def _on_drag_update(self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float) -> None:
        if abs(offset_x) < DRAG_THRESHOLD and abs(offset_y) < DRAG_THRESHOLD:
            return
        self._moved = True
        ok, start_x, start_y = gesture.get_start_point()
        if not ok:
            start_x, start_y = 0.0, 0.0
        native = self.get_native()
        surface = native.get_surface() if native is not None else None
        device = gesture.get_device()
        if surface is None or device is None or not isinstance(surface, Gdk.Toplevel):
            return
        point = Graphene.Point.alloc()
        point.init(start_x, start_y)
        mx, my = start_x, start_y
        if native is not None:
            success, out = self.compute_point(native, point)
            if success and out is not None:
                mx, my = out.x, out.y
        surface.begin_move(device, gesture.get_current_button(), mx, my, gesture.get_current_event_time())
        gesture.reset()

    def _popup_menu(self, x: float, y: float) -> None:
        # The menu itself is data (`menu_model`), built from session state;
        # this method only converts it. What the menu holds -- and the
        # seven-entry bound -- is proven where tests can reach it.
        app = self.get_application()
        menu_def = build_pet_menu(
            listening=bool(getattr(app, "is_listening", lambda: False)()),
            speaking=bool(getattr(app, "is_speaking", lambda: False)()),
            assistant_paused=bool(
                getattr(app, "assistant_paused", lambda: False)()
            ),
            character_paused=self._machine.paused,
        )
        menu = self._to_gio_menu(menu_def.items)
        popover = Gtk.PopoverMenu.new_from_model(menu)
        popover.set_parent(self)
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        popover.set_pointing_to(rect)
        popover.popup()

    @staticmethod
    def _to_gio_menu(items) -> Gio.Menu:
        menu = Gio.Menu()
        for item in items:
            if item.hidden:
                continue
            if item.children:
                menu.append_submenu(item.label, PetWindow._to_gio_menu(item.children))
            else:
                menu.append(item.label, item.action)
        return menu

    def show_window_menu(self) -> None:
        surface = self.get_surface()
        display = self.get_display()
        if surface is None or display is None or not isinstance(surface, Gdk.Toplevel):
            return
        seat = display.get_default_seat()
        device = seat.get_pointer() if seat is not None else None
        if device is None:
            return
        surface.show_window_menu(device, self.get_width() / 2, 8)

    def _on_close(self, *_args: object) -> bool:
        self._persist_position()
        app = self.get_application()
        if app is not None:
            app.activate_action("quit", None)
        return False

    def destroy(self) -> None:  # type: ignore[override]
        if self._tick_id:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = 0
        if self._unsub:
            self._unsub()
        super().destroy()
