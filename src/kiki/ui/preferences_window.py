from __future__ import annotations

from collections.abc import Callable

from kiki.ai.factory import create_provider
from kiki.ai.provider import ProviderHealth
from kiki.config.settings import VALID_TTS_SPEAKERS, Settings, save_settings
from kiki.platform import autostart as autostart_mod
from kiki.runtime.async_bridge import AsyncBridge
from kiki.storage.secrets import OPENAI_API_KEY, SecretStore, SecretStoreError
from kiki.ui.gi_bootstrap import Adw, Gtk
from kiki.voice.system_tts import system_tts_available
from kiki.voice.tts_client import TtsHealth, tts_health


class PreferencesWindow(Adw.PreferencesDialog):
    def __init__(
        self,
        *,
        settings: Settings,
        secrets: SecretStore,
        bridge: AsyncBridge,
        on_change: Callable[[Settings], None],
        on_tts_test: Callable[[], None] | None = None,
        memories: object | None = None,
        routines: object | None = None,
    ) -> None:
        super().__init__()
        self.set_title("KIKI-Einstellungen")
        self._settings = settings
        self._secrets = secrets
        self._bridge = bridge
        self._on_change = on_change
        self._on_tts_test = on_tts_test
        self._memories = memories
        self._memory_group: Adw.PreferencesGroup | None = None
        self._memory_rows: list[Gtk.Widget] = []
        self._routines = routines
        self._routine_group: Adw.PreferencesGroup | None = None
        self._routine_rows: list[Gtk.Widget] = []
        self._status = Gtk.Label(xalign=0, wrap=True)
        self._status.add_css_class("dim-label")

        self.add(self._page_general())
        self.add(self._page_ai())
        self.add(self._page_privacy())
        self.add(self._page_personality())
        if memories is not None:
            self.add(self._page_memory())
        if routines is not None:
            self.add(self._page_routines())

    def _persist(self) -> None:
        save_settings(self._settings)
        self._on_change(self._settings)

    def _page_general(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Allgemein", icon_name="preferences-desktop-display-symbolic")
        pet = Adw.PreferencesGroup(title="Desktop-Pet")
        adj = Gtk.Adjustment(
            value=self._settings.pet.scale,
            lower=0.5,
            upper=2.5,
            step_increment=0.1,
            page_increment=0.25,
        )
        scale = Adw.SpinRow(adjustment=adj, digits=2, title="Größe")
        scale.set_subtitle("Relative Höhe der Figur (ca. 160–650 Pixel).")
        scale.connect("changed", self._on_scale)
        always = Adw.SwitchRow(title="Immer im Vordergrund anfordern")
        always.set_subtitle(
            "Unter GNOME/Wayland kann die App das nicht selbst setzen. "
            "Nutze Alt+Leertaste → Immer im Vordergrund, oder den Eintrag im Pet-Menü."
        )
        always.set_active(self._settings.pet.always_on_top)
        always.connect("notify::active", self._on_always)
        click = Adw.SwitchRow(title="Klickdurchlässiger Hintergrund")
        click.set_subtitle("Transparente Pixel geben Klicks an den Desktop weiter. Die Figur bleibt klickbar.")
        click.set_active(self._settings.pet.click_through_idle)
        click.connect("notify::active", self._on_click)
        greet = Adw.SwitchRow(title="Begrüßung beim Start")
        greet.set_active(self._settings.app.greet_on_start)
        greet.connect("notify::active", lambda row, *_: self._set_greet(row.get_active()))
        auto = Adw.SwitchRow(title="Bei Anmeldung starten")
        auto.set_subtitle("Legt eine XDG-Autostart-Datei in ~/.config/autostart an.")
        auto.set_active(self._settings.app.autostart)
        auto.connect("notify::active", self._on_autostart)
        pet.add(scale)
        pet.add(always)
        pet.add(click)
        pet.add(greet)
        pet.add(auto)
        page.add(pet)
        return page

    def _page_ai(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="KI", icon_name="utilities-terminal-symbolic")
        group = Adw.PreferencesGroup(title="Anbieter")
        models = Gtk.StringList.new(["Ollama (lokal)", "OpenAI-kompatibel (z. B. SpaceXAI)"])
        self._provider_row = Adw.ComboRow(title="Provider", model=models)
        self._provider_row.set_selected(0 if self._settings.ai.provider == "ollama" else 1)
        self._provider_row.connect("notify::selected", self._on_provider)

        self._ollama_url = Adw.EntryRow(title="Ollama-URL")
        self._ollama_url.set_text(self._settings.ai.ollama.base_url)
        self._ollama_url.connect("notify::text", lambda *_: self._save_ai())
        self._ollama_model = Adw.EntryRow(title="Ollama-Modell")
        self._ollama_model.set_text(self._settings.ai.ollama.model)
        self._ollama_model.connect("notify::text", lambda *_: self._save_ai())

        self._oai_url = Adw.EntryRow(title="OpenAI-kompatible Basis-URL")
        self._oai_url.set_text(self._settings.ai.openai_compatible.base_url)
        self._oai_url.connect("notify::text", lambda *_: self._save_ai())
        self._oai_model = Adw.EntryRow(title="Modellname")
        self._oai_model.set_text(self._settings.ai.openai_compatible.model)
        self._oai_model.connect("notify::text", lambda *_: self._save_ai())
        self._api_key = Adw.PasswordEntryRow(title="API-Key (GNOME Keyring)")
        try:
            existing = self._secrets.get(OPENAI_API_KEY)
        except SecretStoreError:
            existing = None
        if existing:
            self._api_key.set_text(existing)
        self._api_key.connect("notify::text", self._on_api_key)

        test = Adw.ButtonRow(title="Verbindung testen")
        test.set_activatable(True)
        test.connect("activated", lambda *_args: self._test_connection())
        test.connect("activate", lambda *_args: self._test_connection())

        for widget in (
            self._provider_row,
            self._ollama_url,
            self._ollama_model,
            self._oai_url,
            self._oai_model,
            self._api_key,
            test,
        ):
            group.add(widget)
        group.add(self._status)
        page.add(group)
        hint = Adw.PreferencesGroup(title="Hinweise")
        hint.set_description(
            "Lokal: `ollama pull qwen3-vl:4b` — kleines Modell mit Deutsch und Bildverständnis. "
            "Mit mehr Speicher liefert `qwen3-vl:8b` meist die bessere Persona- und Coding-Qualität. "
            "Für einen externen Dienst empfehlen wir SpaceXAI (https://api.x.ai/v1, grok-4.5). "
            "Der API-Key landet nur im Secret Service, niemals in der TOML-Datei. "
            "Bilder sendet KIKI nur, wenn du sie im Chat anhängst — kein automatischer Screenshot."
        )
        page.add(hint)
        return page

    def _page_privacy(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Privatsphäre", icon_name="security-medium-symbolic")
        panic = Adw.PreferencesGroup(title="Panic-Schalter")
        row = Adw.SwitchRow(title="Alle Integrationen deaktivieren")
        row.set_subtitle("Sofort. Kein Status, keine Tools, keine Homelab-Aufrufe.")
        row.set_active(self._settings.app.privacy_panic)
        row.connect("notify::active", self._on_panic)
        panic.add(row)
        integ = Adw.PreferencesGroup(title="Integrationen")
        master = Adw.SwitchRow(title="Integrationen erlaubt")
        master.set_active(self._settings.integrations.enabled)
        master.connect(
            "notify::active",
            lambda r, *_: self._set_integ_master(r.get_active()),
        )
        integ.add(master)
        screen = Adw.PreferencesGroup(title="Bildschirm")
        shot = Adw.SwitchRow(title="Bildschirmfoto erlauben")
        shot.set_subtitle("Nur nach Klick und Bestätigung. Danach erscheint der Systemdialog.")
        shot.set_active(self._settings.screenshot.enabled)
        shot.connect("notify::active", lambda r, *_: self._set_shot(r.get_active()))
        interactive = Adw.SwitchRow(title="Bereich im Systemdialog wählen")
        interactive.set_subtitle("Aus: ganzes Bild, soweit der Compositor das zulässt.")
        interactive.set_active(self._settings.screenshot.interactive)
        interactive.connect("notify::active", lambda r, *_: self._set_shot_interactive(r.get_active()))
        screen.add(shot)
        screen.add(interactive)
        voice = Adw.PreferencesGroup(title="Sprache")
        vrow = Adw.SwitchRow(title="Spracheingabe")
        vrow.set_subtitle("Push-to-talk, lokal mit Vosk (Deutsch). Kein Cloud-Dienst.")
        vrow.set_active(self._settings.voice.enabled)
        vrow.connect("notify::active", lambda r, *_: self._set_voice(r.get_active()))
        auto = Adw.SwitchRow(title="Gesprochenes sofort senden")
        auto.set_active(self._settings.voice.auto_send)
        auto.connect("notify::active", lambda r, *_: self._set_voice_auto(r.get_active()))
        wake = Adw.SwitchRow(title="Auf „KIKI“ hören (Weckwort)")
        wake.set_subtitle(
            "Das Mikrofon bleibt offen. Alles bleibt lokal, nichts wird gespeichert — "
            "aber KIKI erkennt dafür laufend Sprache. Nur die Äußerung nach dem "
            "Weckwort wird verwendet, alles andere sofort verworfen."
        )
        wake.set_active(self._settings.voice.wake.enabled)
        wake.connect("notify::active", lambda r, *_: self._set_wake(r.get_active()))
        voice.add(vrow)
        voice.add(auto)
        voice.add(wake)
        tts = Adw.PreferencesGroup(title="Sprachausgabe")
        tts.set_description(
            "Antworten liest ein lokaler Dienst vor: Qwen3-TTS 0.6B CustomVoice auf der GPU. "
            "Wenn er fehlt, übernimmt eine lokale Fedora-Systemstimme."
        )
        trow = Adw.SwitchRow(title="Antworten vorlesen")
        trow.set_subtitle("Satzweise, sobald der Chat Text liefert. Mikrofon bricht das Sprechen ab.")
        trow.set_active(self._settings.tts.enabled)
        trow.connect("notify::active", lambda r, *_: self._set_tts(r.get_active()))
        stream = Adw.SwitchRow(title="Während des Streams sprechen")
        stream.set_subtitle("Aus: erst nach der vollständigen Antwort.")
        stream.set_active(self._settings.tts.stream_sentences)
        stream.connect("notify::active", lambda r, *_: self._set_tts_stream(r.get_active()))
        fallback = Adw.SwitchRow(title="Lokale Ersatzstimme")
        fallback.set_subtitle("Verwendet espeak-ng, wenn der GPU-TTS-Dienst nicht läuft.")
        fallback.set_active(self._settings.tts.fallback_to_system)
        fallback.connect("notify::active", lambda r, *_: self._set_tts_fallback(r.get_active()))
        speakers = Gtk.StringList.new(list(VALID_TTS_SPEAKERS))
        self._tts_speaker = Adw.ComboRow(title="Stimme (CustomVoice)", model=speakers)
        try:
            self._tts_speaker.set_selected(list(VALID_TTS_SPEAKERS).index(self._settings.tts.speaker))
        except ValueError:
            self._tts_speaker.set_selected(list(VALID_TTS_SPEAKERS).index("Serena"))
        self._tts_speaker.connect("notify::selected", self._on_tts_speaker)
        self._tts_url = Adw.EntryRow(title="TTS-Dienst URL")
        self._tts_url.set_text(self._settings.tts.base_url)
        self._tts_url.connect("notify::text", lambda *_: self._save_tts_url())
        ping = Adw.ButtonRow(title="TTS-Dienst prüfen")
        ping.set_activatable(True)
        ping.connect("activated", lambda *_: self._test_tts())
        ping.connect("activate", lambda *_: self._test_tts())
        sample = Adw.ButtonRow(title="Stimme testen")
        sample.set_activatable(True)
        sample.connect("activated", lambda *_: self._run_tts_sample())
        sample.connect("activate", lambda *_: self._run_tts_sample())
        self._tts_status = Gtk.Label(xalign=0, wrap=True)
        self._tts_status.add_css_class("dim-label")
        tts.add(trow)
        tts.add(stream)
        tts.add(fallback)
        tts.add(self._tts_speaker)
        tts.add(self._tts_url)
        tts.add(ping)
        tts.add(sample)
        tts.add(self._tts_status)
        agency = Adw.PreferencesGroup(title="Selbstständigkeit")
        agency.set_description(
            "Steuert, was KIKI von sich aus tun darf. Schreibende und externe "
            "Aktionen zeigen eine Freigabekarte — außer im Jarvis-Modus, der "
            "bewusst ohne Rückfragen handelt."
        )
        use_tools = Adw.SwitchRow(title="KIKI darf Werkzeuge selbst aufrufen")
        use_tools.set_subtitle("Aus: KIKI antwortet nur mit Text und fragt dich nach Daten.")
        use_tools.set_active(self._settings.tools.model_tool_use)
        use_tools.connect("notify::active", lambda r, *_: self._set_model_tool_use(r.get_active()))
        self._autonomy_ids = ["strict", "balanced", "trusted", "jarvis"]
        levels = Gtk.StringList()
        for label in (
            "Nur lesen (strict)",
            "Lesen + Steuerung (balanced)",
            "… + Öffnen ohne Nachfrage (trusted)",
            "… + Alles ohne Rückfragen (jarvis, experimentell)",
        ):
            levels.append(label)
        self._autonomy_row = Adw.ComboRow(title="Vertrauensstufe", model=levels)
        self._autonomy_row.set_subtitle(
            "„trusted“ lässt KIKI Ordner, Dateien, Terminal, Editor und Links selbst "
            "öffnen — nur in registrierten Workspaces. „jarvis“ handelt zusätzlich "
            "Schreibendes und Externes ohne Karte; verbotene Befehle, der "
            "Panic-Schalter und das Audit greifen weiterhin."
        )
        current_level = self._settings.tools.autonomy
        self._autonomy_row.set_selected(
            self._autonomy_ids.index(current_level) if current_level in self._autonomy_ids else 0
        )
        self._autonomy_row.connect("notify::selected", self._on_autonomy)
        agency.add(use_tools)
        agency.add(self._autonomy_row)

        proactive = Adw.PreferencesGroup(title="Von sich aus melden")
        proactive.set_description(
            "KIKI darf auf Akku und Speicherplatz achten und sich melden, wenn "
            "etwas knapp wird. Sie meldet nur — ausführen darf sie weiterhin "
            "nichts von allein."
        )
        watch_on = Adw.SwitchRow(title="Selbst auf Akku und Speicher achten")
        watch_on.set_active(self._settings.watch.enabled)
        watch_on.connect("notify::active", lambda r, *_: self._set_watch(r.get_active()))
        watch_speak = Adw.SwitchRow(title="Warnungen laut aussprechen")
        watch_speak.set_subtitle(
            "Aus: nur eine stille Desktop-Benachrichtigung. Während du mit KIKI "
            "sprichst, bleibt sie ohnehin still."
        )
        watch_speak.set_active(self._settings.watch.speak)
        watch_speak.connect("notify::active", lambda r, *_: self._set_watch_speak(r.get_active()))
        quiet = Adw.EntryRow(title="Ruhezeit von")
        quiet.set_text(self._settings.watch.quiet_start)
        quiet.connect("notify::text", lambda r, *_: self._set_quiet("start", r.get_text()))
        quiet_end = Adw.EntryRow(title="Ruhezeit bis")
        quiet_end.set_text(self._settings.watch.quiet_end)
        quiet_end.connect("notify::text", lambda r, *_: self._set_quiet("end", r.get_text()))
        proactive.add(watch_on)
        proactive.add(watch_speak)
        proactive.add(quiet)
        proactive.add(quiet_end)
        quiet_hint = Gtk.Label(
            label="In der Ruhezeit sagt KIKI nichts. Nur dringende Meldungen "
            "erscheinen dann noch stumm als Benachrichtigung. Format HH:MM; "
            "gleiche Zeiten heißt keine Ruhezeit.",
            xalign=0,
            wrap=True,
        )
        quiet_hint.add_css_class("caption")
        quiet_hint.add_css_class("dim-label")
        proactive.add(quiet_hint)

        page.add(panic)
        page.add(agency)
        page.add(proactive)
        page.add(integ)
        page.add(screen)
        page.add(voice)
        page.add(tts)
        note = Adw.PreferencesGroup()
        note.set_description(
            "KIKI sendet lokale Systemdaten nie automatisch an ein Modell. "
            "Bildschirmfotos und Sprache hängen nur an, wenn du sie auslöst. "
            "Chat-Antworten bleiben lokal (Ollama + Qwen3-TTS). "
            "Werkzeuge muss der Entwickler einzeln für Modellaufrufe freigeben; "
            "schreibende Tools brauchen immer eine Bestätigung. "
            "Beliebige Shell-Befehle sind verboten."
        )
        page.add(note)
        return page

    def _page_memory(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Gedächtnis", icon_name="view-list-symbolic")
        intro = Adw.PreferencesGroup(title="Was KIKI sich gemerkt hat")
        intro.set_description(
            "Diese Einträge gehen bei jeder Antwort in den Systemprompt ein. "
            "Sie entstehen nur, wenn du KIKI ausdrücklich bittest, sich etwas zu "
            "merken, und jeder Eintrag wurde vorher in einer Freigabekarte gezeigt. "
            "Alles liegt lokal in der SQLite-Datei und geht mit dem Panic-Schalter "
            "nicht mehr an das Modell."
        )
        page.add(intro)

        self._memory_group = Adw.PreferencesGroup()
        page.add(self._memory_group)

        actions = Adw.PreferencesGroup()
        clear = Adw.ButtonRow(title="Alles vergessen")
        clear.set_activatable(True)
        clear.add_css_class("destructive-action")
        clear.connect("activated", lambda *_: self._confirm_clear_memories())
        clear.connect("activate", lambda *_: self._confirm_clear_memories())
        actions.add(clear)
        page.add(actions)

        self._reload_memories()
        return page

    def _reload_memories(self) -> None:
        group = self._memory_group
        if group is None or self._memories is None:
            return
        for row in self._memory_rows:
            group.remove(row)
        self._memory_rows = []
        try:
            items = self._memories.list()
        except Exception:
            row = Adw.ActionRow(title="Gedächtnis nicht lesbar")
            group.add(row)
            self._memory_rows.append(row)
            return
        if not items:
            row = Adw.ActionRow(title="Noch nichts gemerkt")
            row.set_subtitle("Sag KIKI zum Beispiel: „Merk dir, dass ich Fedora nutze.“")
            group.add(row)
            self._memory_rows.append(row)
            return
        group.set_title(f"{len(items)} Einträge")
        for item in items:
            row = Adw.ActionRow(title=item.content)
            row.set_subtitle(f"{item.kind} · {item.created_at[:10]}")
            row.set_title_lines(0)
            button = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
            button.add_css_class("flat")
            button.set_tooltip_text("Diese Erinnerung löschen")
            button.connect("clicked", lambda _b, mid=item.id: self._delete_memory(mid))
            row.add_suffix(button)
            group.add(row)
            self._memory_rows.append(row)

    def _delete_memory(self, memory_id: str) -> None:
        if self._memories is None:
            return
        try:
            self._memories.delete(memory_id)
        except Exception:
            pass
        self._reload_memories()

    def _confirm_clear_memories(self) -> None:
        if self._memories is None:
            return
        dialog = Adw.AlertDialog(
            heading="Alles vergessen?",
            body=(
                "Alle gemerkten Einträge werden endgültig gelöscht. "
                "Chatverlauf und Einstellungen bleiben erhalten."
            ),
        )
        dialog.add_response("cancel", "Abbrechen")
        dialog.add_response("clear", "Alles löschen")
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.set_response_appearance("clear", Adw.ResponseAppearance.DESTRUCTIVE)

        def _done(_dialog: Adw.AlertDialog, response: str) -> None:
            if response != "clear":
                return
            try:
                self._memories.clear()
            except Exception:
                pass
            self._reload_memories()

        dialog.connect("response", _done)
        dialog.present(self)

    def _page_routines(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Routinen", icon_name="emblem-synchronizing-symbolic")
        intro = Adw.PreferencesGroup(title="Wenn-Dann-Routinen")
        intro.set_description(
            "Jede Routine wurde als komplettes Rezept bestätigt: Auslöser, "
            "Werkzeug und Argumente. Sie feuert später genau so und ohne "
            "erneute Frage — der Panic-Schalter hält sie jederzeit an."
        )
        page.add(intro)

        self._routine_group = Adw.PreferencesGroup()
        page.add(self._routine_group)

        self._reload_routines()
        return page

    def _reload_routines(self) -> None:
        group = self._routine_group
        if group is None or self._routines is None:
            return
        for row in self._routine_rows:
            group.remove(row)
        self._routine_rows = []
        try:
            items = self._routines.list()
        except Exception:
            row = Adw.ActionRow(title="Routinen nicht lesbar")
            group.add(row)
            self._routine_rows.append(row)
            return
        if not items:
            row = Adw.ActionRow(title="Noch keine Routinen")
            row.set_subtitle(
                "Sag KIKI zum Beispiel: „Wenn der Akku unter 15 Prozent fällt, "
                "leg eine Notiz an.“"
            )
            group.add(row)
            self._routine_rows.append(row)
            return
        group.set_title(f"{len(items)} Routinen")
        for routine in items:
            row = Adw.ActionRow(title=routine.name)
            row.set_subtitle(
                f"{routine.trigger.describe()} → {routine.tool_name} · "
                f"{routine.fired_count}× gefeuert"
            )
            row.set_title_lines(0)
            switch = Gtk.Switch(
                active=routine.enabled, valign=Gtk.Align.CENTER, tooltip_text="Routine aktiv"
            )

            def _on_state_set(_switch: Gtk.Switch, state: bool, rid: str = routine.id) -> bool:
                self._toggle_routine(rid, bool(state))
                # False lets the switch keep its new visual state.
                return False

            switch.connect("state-set", _on_state_set)
            row.add_suffix(switch)
            button = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
            button.add_css_class("flat")
            button.set_tooltip_text("Diese Routine löschen")
            button.connect("clicked", lambda _b, rid=routine.id: self._delete_routine(rid))
            row.add_suffix(button)
            group.add(row)
            self._routine_rows.append(row)

    def _toggle_routine(self, routine_id: str, enabled: bool) -> None:
        if self._routines is None:
            return
        try:
            self._routines.set_enabled(routine_id, enabled)
        except Exception:
            pass

    def _delete_routine(self, routine_id: str) -> None:
        if self._routines is None:
            return
        try:
            self._routines.delete(routine_id)
        except Exception:
            pass
        self._reload_routines()

    def _page_personality(self) -> Adw.PreferencesPage:
        from kiki.ai.persona import (
            CUSTOM_ID,
            PERSONAS,
            looks_like_legacy_full_prompt,
        )

        page = Adw.PreferencesPage(title="Persönlichkeit", icon_name="user-available-symbolic")

        choose = Adw.PreferencesGroup(title="Ton")
        choose.set_description(
            "Bestimmt, wie KIKI klingt. Ihre Regeln zu Wahrheit, Werkzeugen und "
            "Freigaben bleiben in jedem Ton gleich — sie stehen unten und lassen "
            "sich nicht überschreiben."
        )
        self._persona_ids = [p.id for p in PERSONAS] + [CUSTOM_ID]
        names = Gtk.StringList()
        for persona in PERSONAS:
            names.append(persona.name)
        names.append("Eigene")
        self._persona_row = Adw.ComboRow(title="Persona", model=names)
        current = self._settings.persona.id
        self._persona_row.set_selected(
            self._persona_ids.index(current) if current in self._persona_ids else 0
        )
        self._persona_row.set_subtitle(self._persona_subtitle())
        self._persona_row.connect("notify::selected", self._on_persona)
        choose.add(self._persona_row)

        address = Adw.EntryRow(title="Anrede")
        address.set_text(self._settings.persona.address)
        address.connect("notify::text", lambda r, *_: self._set_address(r.get_text()))
        choose.add(address)
        hint = Gtk.Label(
            label="Leer lassen, wenn KIKI dich nicht mit Namen ansprechen soll. "
            "Sonst zum Beispiel dein Vorname.",
            xalign=0,
            wrap=True,
        )
        hint.add_css_class("caption")
        hint.add_css_class("dim-label")
        choose.add(hint)
        page.add(choose)

        self._prompt_group = Adw.PreferencesGroup(title="Eigener Text")
        self._prompt = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._prompt.get_buffer().set_text(self._settings.persona_prompt())
        self._prompt.get_buffer().connect("changed", self._buffer_changed)
        scroll = Gtk.ScrolledWindow(min_content_height=220, hexpand=True, vexpand=True)
        scroll.set_child(self._prompt)
        self._prompt_group.add(scroll)
        self._legacy_note = Gtk.Label(xalign=0, wrap=True)
        self._legacy_note.add_css_class("caption")
        self._legacy_note.add_css_class("warning")
        self._prompt_group.add(self._legacy_note)
        if looks_like_legacy_full_prompt(self._settings.ai.system_prompt):
            self._legacy_note.set_text(
                "Dein Text enthält noch Regeln, die KIKI inzwischen selbst mitbringt "
                "(Wahrheit, Werkzeuge, Freigaben). Doppelt schadet nicht, aber ein "
                "Preset oben ist jetzt kürzer und bleibt automatisch aktuell."
            )
        else:
            self._legacy_note.set_visible(False)
        page.add(self._prompt_group)
        self._sync_persona_editor()

        core = Adw.PreferencesGroup(title="Feste Regeln")
        core.set_description(
            "Diese Regeln hängen immer an KIKIs Prompt, in jedem Ton. Sie kommen "
            "aus dem Programm und werden nicht in deiner Konfiguration gespeichert, "
            "damit sie mit jedem Update aktuell bleiben."
        )
        from kiki.ai.persona import CORE_RULES

        view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, editable=False, cursor_visible=False)
        view.get_buffer().set_text(CORE_RULES)
        view.add_css_class("dim-label")
        core_scroll = Gtk.ScrolledWindow(min_content_height=200, hexpand=True)
        core_scroll.set_child(view)
        core.add(core_scroll)
        page.add(core)
        return page

    def _persona_subtitle(self) -> str:
        from kiki.ai.persona import get_persona

        persona = get_persona(self._settings.persona.id)
        return persona.description if persona else "Dein eigener Text steuert den Ton."

    def _sync_persona_editor(self) -> None:
        """The text box only edits something when the custom persona is active."""
        from kiki.ai.persona import CUSTOM_ID

        custom = self._settings.persona.id == CUSTOM_ID
        self._prompt.set_editable(custom)
        self._prompt.set_cursor_visible(custom)
        self._prompt_group.set_title("Eigener Text" if custom else "Text dieses Presets")
        self._prompt_group.set_description(
            "Wird als KIKIs Persönlichkeit verwendet."
            if custom
            else "Nur zur Ansicht. Wähle „Eigene“, um selbst zu schreiben."
        )

    def _on_persona(self, row: Adw.ComboRow, *_args: object) -> None:
        from kiki.ai.persona import CUSTOM_ID, get_persona

        index = int(row.get_selected())
        if not 0 <= index < len(self._persona_ids):
            return
        chosen = self._persona_ids[index]
        previous = self._settings.persona.id
        if chosen == CUSTOM_ID and previous != CUSTOM_ID:
            # Start the custom text from what was on screen, so switching to
            # "Eigene" never drops the user into an empty box.
            preset = get_persona(previous)
            if preset is not None and not self._settings.ai.system_prompt.strip():
                self._settings.ai.system_prompt = preset.prompt
        self._settings.persona.id = chosen
        row.set_subtitle(self._persona_subtitle())
        buffer = self._prompt.get_buffer()
        buffer.handler_block_by_func(self._buffer_changed)
        buffer.set_text(self._settings.persona_prompt())
        buffer.handler_unblock_by_func(self._buffer_changed)
        self._sync_persona_editor()
        self._persist()

    def _set_address(self, value: str) -> None:
        self._settings.persona.address = " ".join(str(value).split())[:60]
        self._persist()

    def _on_scale(self, row: Adw.SpinRow) -> None:
        self._settings.pet.scale = float(row.get_value())
        self._persist()

    def _on_always(self, row: Adw.SwitchRow, *_args: object) -> None:
        self._settings.pet.always_on_top = bool(row.get_active())
        self._persist()

    def _on_click(self, row: Adw.SwitchRow, *_args: object) -> None:
        self._settings.pet.click_through_idle = bool(row.get_active())
        self._persist()

    def _set_greet(self, value: bool) -> None:
        self._settings.app.greet_on_start = value
        self._persist()

    def _on_autostart(self, row: Adw.SwitchRow, *_args: object) -> None:
        enabled = bool(row.get_active())
        self._settings.app.autostart = enabled
        autostart_mod.set_enabled(enabled)
        self._persist()

    def _on_panic(self, row: Adw.SwitchRow, *_args: object) -> None:
        self._settings.app.privacy_panic = bool(row.get_active())
        self._persist()

    def _set_integ_master(self, value: bool) -> None:
        self._settings.integrations.enabled = value
        self._persist()

    def _set_watch(self, value: bool) -> None:
        self._settings.watch.enabled = value
        self._persist()

    def _set_watch_speak(self, value: bool) -> None:
        self._settings.watch.speak = value
        self._persist()

    def _set_quiet(self, which: str, value: str) -> None:
        from datetime import time as _time

        from kiki.watch.notifier import parse_clock

        text = str(value).strip()
        fallback = _time(22, 0) if which == "start" else _time(8, 0)
        # Keep whatever the user typed only if it parses; otherwise the stored
        # value stays usable and the notifier cannot end up with a broken window.
        parsed = parse_clock(text, fallback)
        normalized = f"{parsed.hour:02d}:{parsed.minute:02d}"
        if which == "start":
            self._settings.watch.quiet_start = normalized
        else:
            self._settings.watch.quiet_end = normalized
        self._persist()

    def _set_model_tool_use(self, value: bool) -> None:
        self._settings.tools.model_tool_use = value
        self._persist()

    def _on_autonomy(self, row: Adw.ComboRow, *_args: object) -> None:
        index = int(row.get_selected())
        if 0 <= index < len(self._autonomy_ids):
            self._settings.tools.autonomy = self._autonomy_ids[index]
            self._persist()

    def _set_shot(self, value: bool) -> None:
        self._settings.screenshot.enabled = value
        self._persist()

    def _set_shot_interactive(self, value: bool) -> None:
        self._settings.screenshot.interactive = value
        self._persist()

    def _set_voice(self, value: bool) -> None:
        self._settings.voice.enabled = value
        self._persist()

    def _set_wake(self, value: bool) -> None:
        self._settings.voice.wake.enabled = value
        self._persist()

    def _set_voice_auto(self, value: bool) -> None:
        self._settings.voice.auto_send = value
        self._persist()

    def _set_tts(self, value: bool) -> None:
        self._settings.tts.enabled = value
        self._persist()

    def _set_tts_stream(self, value: bool) -> None:
        self._settings.tts.stream_sentences = value
        self._persist()

    def _set_tts_fallback(self, value: bool) -> None:
        self._settings.tts.fallback_to_system = value
        self._persist()

    def _on_tts_speaker(self, row: Adw.ComboRow, *_args: object) -> None:
        idx = int(row.get_selected())
        if 0 <= idx < len(VALID_TTS_SPEAKERS):
            self._settings.tts.speaker = VALID_TTS_SPEAKERS[idx]
            self._persist()

    def _save_tts_url(self) -> None:
        text = self._tts_url.get_text().strip().rstrip("/")
        try:
            from kiki.config.settings import settings_from_mapping

            mapping = self._settings.to_mapping()
            mapping["tts"]["base_url"] = text
            settings_from_mapping(mapping)
        except Exception as exc:
            self._tts_status.set_text(str(exc))
            return
        self._settings.tts.base_url = text
        self._persist()

    def _test_tts(self) -> None:
        self._save_tts_url()
        self._tts_status.set_text("Prüfe TTS-Dienst …")
        url = self._settings.tts.base_url

        def _ok(health: TtsHealth) -> None:
            if health.ok and not health.ready:
                self._tts_status.set_text(health.detail + " · Modell wird noch geladen.")
                return
            if (
                not health.ok
                and self._settings.tts.fallback_to_system
                and system_tts_available()
            ):
                self._tts_status.set_text(
                    f"{health.detail}. Lokale Ersatzstimme ist bereit."
                )
                return
            extra = ""
            if health.device:
                extra = f" ({health.device})"
            prefix = "OK. " if health.ok else ""
            dummy = " Dummy-Modus." if health.dummy else ""
            self._tts_status.set_text(prefix + health.detail + extra + dummy)

        def _err(exc: BaseException) -> None:
            self._tts_status.set_text(str(exc))

        self._bridge.submit(tts_health(url), on_success=_ok, on_error=_err)

    def _run_tts_sample(self) -> None:
        self._save_tts_url()
        if self._on_tts_test is None:
            self._tts_status.set_text("Stimme testen ist hier nicht verfügbar.")
            return
        self._tts_status.set_text("Spreche Testsatz …")
        self._on_tts_test()
        self._tts_status.set_text("Testsatz gestartet. Mit „Sprechen beenden“ jederzeit abbrechbar.")

    def _on_provider(self, row: Adw.ComboRow, *_args: object) -> None:
        self._settings.ai.provider = "ollama" if row.get_selected() == 0 else "openai_compatible"
        self._persist()

    def _save_ai(self) -> None:
        self._settings.ai.ollama.base_url = self._ollama_url.get_text().strip().rstrip("/")
        self._settings.ai.ollama.model = self._ollama_model.get_text().strip()
        self._settings.ai.openai_compatible.base_url = self._oai_url.get_text().strip().rstrip("/")
        self._settings.ai.openai_compatible.model = self._oai_model.get_text().strip()
        try:
            from kiki.config.settings import settings_from_mapping

            settings_from_mapping(self._settings.to_mapping())
        except Exception as exc:
            self._status.set_text(str(exc))
            return
        self._persist()

    def _buffer_changed(self, _buffer: Gtk.TextBuffer) -> None:
        self._save_prompt()

    def _save_prompt(self) -> None:
        from kiki.ai.persona import CUSTOM_ID

        # Presets are shown read-only, so a change here is always the user's own
        # text. Guarding anyway keeps a programmatic set_text from overwriting it.
        if self._settings.persona.id != CUSTOM_ID:
            return
        buf = self._prompt.get_buffer()
        self._settings.ai.system_prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        self._persist()

    def _on_api_key(self, row: Adw.PasswordEntryRow, *_args: object) -> None:
        try:
            self._secrets.set(OPENAI_API_KEY, row.get_text())
            self._status.set_text("API-Key im GNOME Keyring gespeichert.")
        except SecretStoreError as exc:
            self._status.set_text(str(exc))

    def _test_connection(self) -> None:
        self._save_ai()
        provider = create_provider(self._settings, self._secrets)
        model = (
            self._settings.ai.ollama.model
            if self._settings.ai.provider == "ollama"
            else self._settings.ai.openai_compatible.model
        )
        self._status.set_text("Prüfe Verbindung …")

        def _ok(health: ProviderHealth) -> None:
            extra = ""
            if health.models:
                extra = " Modelle: " + ", ".join(health.models[:12])
            prefix = "OK. " if health.ok else ""
            self._status.set_text(prefix + health.detail + extra)

        def _err(exc: BaseException) -> None:
            self._status.set_text(str(exc))

        self._bridge.submit(provider.ping(model), on_success=_ok, on_error=_err)
