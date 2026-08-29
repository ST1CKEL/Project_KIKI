# KIKI – Architektur, Risiken, MVP

## 1. Architekturentscheidung

KIKI ist eine **GTK4/libadwaita-Anwendung** mit einem **GI-freien Kern**. Die Figur ist ein Zustandsautomat plus austauschbares Asset-Paket, nicht ein fest verdrahtetes Bild.

```
┌──────────────────────────────────────────────────────────┐
│  GTK4 / libadwaita  (Pet, Chat, Statusbar, ConfirmModal) │
│  Keine Business-Logik in Callbacks — nur Darstellung     │
│  Kein PyTorch / kein CUDA im GUI-Prozess                 │
└────────────┬─────────────────────────────┬───────────────┘
             │ GLib.idle_add               │ GActions
             ▼                             ▼
┌─────────────────────┐         ┌─────────────────────────┐
│ AsyncBridge         │         │ EventBus                │
│ (asyncio-Thread)    │         │ chat.stream.*           │
└──────────┬──────────┘         └────────────▲────────────┘
           ▼                                 │
┌────────────────────────────────────────────┴────────────┐
│ ChatService  · CharacterStateMachine  · SpeechDirector  │
│ HarnessSession (AgentRunner, Tools)   · ToolExecutor    │
│ LLMProvider  · ChatRepository         · NotesWorkspace  │
└────────┬──────────────┬──────────────────┬──────────────┘
         ▼              ▼                  ▼
   Ollama            Qwen3-TTS-Dienst    PipeWire
   qwen3-vl:4b       0.6B CustomVoice    pulsesink / pw-play
   Chat + Vision     CUDA / GPU          Audio-Ausgabe
   SQLite WAL        127.0.0.1:18765
```

### Feste Entscheidungen

| Thema | Entscheidung |
|---|---|
| GUI | GTK 4.22 + libadwaita 1.9, `Adw.Application` (Single-Instance über D-Bus-Name) |
| Pet-Fenster | Unverziertes `Gtk.Window` mit transparentem CSS, **nicht** `Adw.ApplicationWindow` |
| Chat/Einstellungen | `Adw.ApplicationWindow` / `Adw.PreferencesDialog` |
| Async | Eigener asyncio-Thread + `GLib.idle_add`. Nicht die experimentelle `GLibEventLoopPolicy`. |
| Config | TOML, XDG (`~/.config/kiki/config.toml`), Defaults im Paket |
| Speicher | SQLite WAL unter `~/.local/share/kiki/` |
| Secrets | nur libsecret / GNOME Keyring. Kein Datei-Fallback. |
| KI-Default | **Ollama** lokal mit `qwen3-vl:4b` (Deutsch + Vision, ~3,3 GB). `qwen3-vl:8b` ist das optionale Qualitätsprofil; der kleinere Default verhindert erzwungene Modell-Downloads bei Updates. Optional OpenAI-kompatibel; empfohlener externer Dienst: **SpaceXAI** (`https://api.x.ai/v1`, `grok-4.5`). |
| TTS | Eigener Dienst `kiki-tts` mit **Qwen3-TTS-12Hz-0.6B-CustomVoice** auf CUDA. Default-Stimme **Serena**, Sprache **German**. GTK spielt WAV über PipeWire; Syntheseaufträge sind abbrechbar und verspätete WAVs werden verworfen. Loopback only. |
| Tools | Default Deny. Unbekannte Namen und eine Hard-Deny-Liste (`run_shell`, `sudo`, …) laufen nie. |
| Modell-Tool-Use | **An** (Phase 2A). Nur Tools mit `model_callable = true` sind für das Modell sichtbar; Policy, Freigabekarte und Audit bleiben unverändert auf dem Pfad. |
| Vertrauensstufe | `tools.autonomy`: `strict` (lesen), `balanced` (Default: + deklarierte Steuerung), `trusted` (+ Öffnen in registrierten Workspaces), `jarvis` (opt-in: alles unbeaufsichtigt, auch Schreiben/Extern — Phase 3A). Außerhalb `jarvis` fragen Schreiben und Extern auf jeder Stufe. |
| Charakter | `renderer = "frames"` in `manifest.toml`. Später lottie/spine/live2d ohne Chat-Umbau. |
| App-ID | `io.github.projectkiki.Kiki` |
| Lizenz | MIT |

Schichten dürfen nicht übersprungen werden: GTK-Callbacks rufen `Gio.Action`s oder den `ChatService` auf, niemals `httpx` oder `sqlite3` direkt.

## 2. Risikoanalyse Fedora 44 / GNOME / Wayland

GNOME  auf Fedora 44 nutzt Wayland. Das ist die größte technische Grenze eines Desktop-Pets.

| Risiko | Wirkung | Mitigation im MVP |
|---|---|---|
| **Keine programmatische Fensterposition** | `move()` gibt es in GTK4 nicht. Wayland verbietet Clients, Koordinaten zu setzen. Untere rechte Ecke ist **nicht garantiert**. | Interaktives Verschieben über `Gdk.Toplevel.begin_move`. GNOME kann die letzte Position der App-ID merken. Setting `pet.anchor` ist ein Hinweis für X11 und die UI, kein Befehl an Mutter. |
| **Kein programmatisches Always-on-top** | GTK4 hat kein `set_keep_above`. Wayland hat keine App-API für den Fensterstapel (GNOME-Diskurs 2025). | Setting wird gespeichert. Auf X11: `_NET_WM_STATE_ABOVE`. Auf Wayland: Pet-Menü „Fenstermenü“ → `show_window_menu`, plus Hinweis auf Alt+Leertaste. |
| **Kein wlr-layer-shell auf GNOME** | `gtk4-layer-shell` ist für Sway/Hyprland. Mutter implementiert das Protokoll nicht. | Bewusst **kein** Layer-Shell-Zwang. Optionales späteres Backend, wenn ein Compositor es anbietet. |
| **Transparenz** | Adwaita malt `window.background` undurchsichtig. | Pet-Fenster entfernt die CSS-Klasse `background`, CSS `background-color: transparent`. PNG mit Alpha. |
| **Klickdurchlässigkeit** | Volles Click-through würde KIKI unbedienbar machen (kein Rechtsklick). | Default: Input-Region aus dem Alpha-Kanal. Transparente Pixel gehen durch, die Figur bleibt klickbar. |
| **Fokusklau beim Start** | Overlay würde die Tastatur stehlen. | `set_focus_on_map(False)`. |
| **Pet in der Übersicht/Dash** | Wayland-Apps können sich nicht zuverlässig aus der Taskleiste nehmen. | Akzeptiert und dokumentiert. |
| **XWayland-Fallback** | `GDK_BACKEND=x11` gibt Position/Always-on-top, verliert aber native Wayland-Integration. | Optional, nicht Default. |
| **UPower/NM über System-D-Bus** | Auf manchen VMs kein Akku, NM kann fehlen. | Snapshot mit `available=false`, keine Exception in die UI. |
| **Secret Service** | Ohne Keyring kein API-Key. | Klare Fehlermeldung, **kein** Klartext in `config.toml`. |
| **Ollama nicht installiert** | Chat schlägt fehl. | Ping im Einstellungsdialog, Fehlerbubble im Chat, Figur wechselt in `error`. |

### Was Wayland *kann*

- Transparentes, undekoriertes Fenster
- PNG-Alpha
- Input-Region
- Interaktives Verschieben (`begin_move`)
- Separates Chatfenster, Toasts, Dialoge
- Single-Instance über `Gio.Application`

## 3. MVP-Plan (umgesetzt)

1. Projektgerüst, XDG, TOML, Logging
2. Zustandsautomat + Frame-Renderer + Charakterpaket KIKI
3. Transparentes Pet, Drag, Kontextmenü, Idle/Thinking/Speaking
4. Chatfenster, Streaming, Markdown, SQLite
5. `OllamaProvider` + `OpenAICompatibleProvider`
6. Read-only Integrationen hinter Tool-Policy und Panic-Schalter
7. Tests für Provider, State-Machine, Config, DB, Policy
8. README + Desktop/Metainfo/systemd-User-Unit

## 4. Was läuft / vorbereitet / Limit

### Läuft im MVP

- Start als GTK4/libadwaita-App
- Verschiebbares Desktop-Pet mit Original-Figur KIKI
- States: idle (+Blink/Sway), thinking, speaking, greet, listening, happy, surprised, sleeping, error, notification
- Chat mit Enter / Shift+Enter, Streaming, Markdown, Code-Kopieren
- Ollama-Streaming, konfigurierbare URL und Modell
- OpenAI-kompatibler Provider inkl. Keyring
- Verbindungstest
- SQLite-Verlauf, neuer Chat, Chat löschen
- Statuskarte auf Klick; dieselben `status_*`-Tools erreicht auch der Agent-Loop
- Default-Deny-Tool-Policy + Audit-Log
- Autostart (XDG)
- Panic-Schalter
- Bildschirmfoto nach Freigabe (xdg-desktop-portal, Fallback Spectacle) an das lokale Vision-Modell
- Spracheingabe Push-to-talk, lokal Vosk Deutsch
- Weckwort „KIKI" (Phase 2B), opt-in und per Default aus
- Sprachausgabe: lokaler Qwen3-TTS-Dienst (0.6B CustomVoice, GPU), Satz-Streaming, PipeWire, Barge-in sowie Abbruch laufender Synthese
- Coding-Workspace-Allowlist (Git-Root, Symlink-Schutz, SQLite-Registry)
- OpenCode-Adapter, verbindliche Plan-First-Sessions mit `plan_session_id`, observe/develop-Policy, Approvals, Prozessgruppen-Runner
- Coding-Session-Fenster und Workspace-Manager (libadwaita)
- Terminal/Datei/https nach Freigabe; Chat-Übergabe in die Coding-Session
- Eigenes PC-Steuerungsfenster: Ordner, Datei, Terminal, Editor, http(s), Zwischenablage und lokale Benachrichtigung – jeweils mit gebundener Einzelfreigabe
- Live-Diff, Editor-Allowlist, Briefing, Chat-Zusammenfassung; Podman/SSH-Stubs fail closed
- Agent-Loop: das Modell ruft `status_*` selbst auf, Ergebnis fließt in die Antwort zurück (Phase 2A)
- Gedächtnis: bestätigte Erinnerungen im Systemprompt, in den Einstellungen löschbar (Phase 2C)
- Umschaltbare Persona, feste Regeln davon getrennt (Phase 2D)
- Proaktive Meldungen zu Akku und Speicher, mit Ruhezeiten (Phase 2E)
- Desktop-Steuerung: MPRIS-Medien, Lautstärke (pactl), Helligkeit (GNOME/KDE-Kaskade), App-Start über .desktop-Einträge, Bildschirm sperren (Phase 3A)
- Jarvis-Modus `tools.autonomy = "jarvis"`: unbeaufsichtigte Ausführung auf allen Risikostufen, opt-in (Phase 3A)
- Freigegebene Wenn-Dann-Routinen (Akku/Speicher → Werkzeugaufruf), in den Einstellungen verwaltbar (Phase 3B)
- System & Netzwerk: WLAN-Gerät schalten, Netzwerke anzeigen, VPN verbinden/trennen (NetworkManager), Ruhezustand/Neustart/Ausschalten (logind) (Phase 3C)

### Phase 2A — Agent-Loop

`ChatService.send()` wählt pro Zug zwischen dem bisherigen Ein-Weg-Stream und
`AgentLoop`. Der Loop läuft, wenn `tools.model_tool_use` gesetzt ist, der Panic-
Schalter aus ist, der Provider `stream_chat_tools` beherrscht und mindestens ein
Tool übrig bleibt. Sonst fällt er auf den alten Pfad zurück — eine fehlende
Tool-Fähigkeit macht KIKI also nicht kaputt, nur wortkarger.

Grenzen des Loops:

- `tools.max_steps` (Default 6) Modellrunden pro Zug; danach ein sichtbarer
  Fehler statt einer halben Antwort.
- `tools.max_tool_calls` (Default 12) Aufrufe pro Zug.
- Ein identischer Aufruf im selben Zug wird nicht erneut ausgeführt, sondern aus
  dem ersten Ergebnis beantwortet.
- Tool-Ergebnisse werden auf 8 KiB gekürzt, bevor sie zurück ins Modell gehen.
- Kaputtes Argument-JSON wird als Tool-Fehler zurückgemeldet, nie geraten.

Der Freigabedialog läuft über `AsyncBridge.ask_ui`: Der Loop wartet auf dem
asyncio-Thread, der Dialog erscheint auf dem GTK-Thread. Eine unbeantwortete
Frage verfällt nach fünf Minuten als **Ablehnung**.

### Phase 2B — Weckwort „KIKI"

`kiki/voice/wake.py` hält einen eigenen GStreamer-`appsink` offen und schiebt
16-kHz-Mono-PCM durch einen Vosk-Erkenner. Der `WakeWordListener` ist ein
Zwei-Zustands-Automat:

| Zustand | Verhalten |
|---|---|
| `waiting` | Jede fertige Äußerung wird gegen die Weckwörter geprüft und **verworfen**. Text verlässt das Modul nicht. |
| `capturing` | Die nächste fertige Äußerung wird als gesprochener Befehl übergeben, danach zurück auf `waiting`. |

Damit braucht der Befehl nach dem Weckwort weder eine zweite Pipeline noch eine
geratene Aufnahmedauer: Vosk beendet eine Äußerung rund 1,0–1,15 s nach dem
Verstummen.

Nach einer final ausgegebenen Antwort darf `FollowUpTurn` denselben
`capturing`-Zustand genau einmal ohne neues Weckwort aktivieren. Dafür müssen
Start per Weckwort, terminaler Run und tatsächlich ausgelieferte Antwort
zusammenkommen. Bestätigungsansagen und andere TTS-Ausgaben erfüllen diese
Bedingung nicht. Ein Timeout beendet die sichtbare Hörphase und führt zurück zu
`waiting`; das Verhalten lässt sich unabhängig vom Weckwortschalter abschalten.

**Warum kein Grammatikmodus.** Der naheliegende Ansatz — `vosk_recognizer_new_grm`
mit `["kiki", "hey kiki", …, "[unk]"]` — wurde gemessen und verworfen. Das kleine
deutsche Modell bildet beliebige Sprache auf die kurzen Weckphrasen ab: „der
schlüssel liegt auf dem tisch" wurde zu `hey kiki`, „der key ist abgelaufen" zu
`kiki`. Ergebnis: **6 Fehlalarme bei 12 Negativsätzen**. Dasselbe Korpus durch
den offenen Erkenner ergab die tatsächlichen Sätze und **0 Fehlalarme**.

Der Preis dieser Entscheidung ist ehrlich zu benennen: Im Wartezustand läuft
lokale Spracherkennung mit vollem Vokabular. Deshalb gelten harte Regeln:

- Audio wird nie auf Platte geschrieben und verlässt den Prozess nie.
- Erkannter Text im Wartezustand verlässt `wake.py` nicht — er wird geprüft und
  verworfen, nie gespeichert, nie geloggt, nie an ein Modell gesendet.
- Opt-in, Default aus. `voice_allowed()` faltet den Panic-Schalter ein.
- Während KIKI spricht, ist der Listener pausiert, sonst weckt ihre eigene
  Stimme sie. Push-to-talk pausiert ihn ebenfalls.

### Phase 2C — Gedächtnis

`MemoryRepository` ist nicht länger nur Schema. Erinnerungen entstehen über drei
Werkzeuge (`kiki/tools/memory_tools.py`), landen in SQLite und gehen als
Datenblock in den Systemprompt.

| Werkzeug | Risiko | Wirkung |
|---|---|---|
| `memory_remember` | WRITE | Freigabekarte in **jeder** Vertrauensstufe |
| `memory_recall` | READ | läuft unbeaufsichtigt |
| `memory_forget` | WRITE | Freigabekarte in **jeder** Vertrauensstufe |

Schreiben ist bewusst bestätigungspflichtig: Was KIKI sich merkt, prägt jede
spätere Antwort, also darf sie das nicht still entscheiden. Die Karte zeigt den
exakten Wortlaut, der gespeichert wird — der kann von dem abweichen, was der
Nutzer gesagt hat.

**Das ist die einzige automatische Prompt-Anreicherung in KIKI.** Sie ist mit
„Automatisches Anreichern von Prompts mit Systemdaten" (weiter unten bewusst
ausgeschlossen) nicht zu verwechseln: Der Block enthält ausschließlich Sätze,
die der Nutzer selbst freigegeben hat, und ist in den Einstellungen unter
**Gedächtnis** vollständig einsehbar und löschbar.

Härtung:

- `clean_content()` entfernt Zeilenumbrüche und Steuerzeichen. Eine Erinnerung
  kann damit die Prompt-Struktur nicht nachbauen — sie bleibt ein Listenpunkt.
- Der Block ist ausdrücklich als Daten gerahmt; der Systemprompt nennt
  „Gemerktes" in derselben Reihe wie Bilder, Logs und Statusblöcke.
- Höchstens 400 Zeichen je Eintrag, 200 Einträge gesamt, 40 Einträge und
  2000 Zeichen im Prompt. Duplikate werden abgelehnt.
- Panic und „Integrationen aus" halten den Block aus dem Prompt **und**
  entfernen die Werkzeuge aus der Sicht des Modells.

### Phase 2D — Persona und Prompt-Aufteilung

Der Systemprompt ist nicht länger ein Block. `kiki/ai/persona.py` trennt ihn:

| Hälfte | Wo | Änderbar |
|---|---|---|
| `CORE_RULES` | im Paket | nein, nur einsehbar |
| Persona | Preset oder `ai.system_prompt` | ja |

`Settings.compose_prompt()` fügt beides zusammen — Persona zuerst, Kern zuletzt,
dazwischen optional die Anrede aus `persona.address`.

**Das behebt einen echten Defekt.** `ai.system_prompt` war ein Skalar; eine
Nutzerkonfiguration überschrieb damit den Paket-Default *vollständig*. Wer KIKI
vor Phase 2A installiert hatte, bekam die später ergänzten Werkzeug- und
Gedächtnisregeln nie zu sehen — der Prompt in `config.toml` war eingefroren.
Seit der Trennung liefert das Paket die Regeln bei jedem Start neu, und die
Persona ist das Einzige, was aus der Konfiguration kommt.

Migration: Eine Konfiguration mit `ai.system_prompt`, aber ohne `[persona]`,
wird als Persona `eigene` gelesen. Der selbst geschriebene Text bleibt also
erhalten und bekommt die aktuellen Regeln angehängt. Ein alter Volltext enthält
die Regeln dann doppelt; die Oberfläche weist darauf hin, schreibt aber nichts um.

Presets: `begleiterin` (Default, bisheriger Ton), `assistenz` (trocken,
vorausschauend), `knapp` (Minimum), `eigene` (freier Text).

#### Kontextfenster und denkende Modelle

Beim Vergleich der Personas fiel auf, dass `qwen3-vl:4b` gelegentlich **leer**
antwortete. Ursache: Das Modell hat einen Thinking-Modus, `message.thinking`
wuchs auf über 13.000 Zeichen, `message.content` blieb leer und der Lauf endete
mit `done_reason: length`. Ollamas `num_ctx`-Default von 4096 war aufgebraucht,
bevor eine Antwort entstand. `think: false` wird von dieser Kombination
akzeptiert, aber ignoriert.

Zwei Konsequenzen im Code:

- `ai.ollama.num_ctx` (Default 8192) wird jetzt gesetzt. Vorher setzte KIKI die
  Option nie und lief immer auf 4096.
- Ein Stream, der ohne Text und mit `done_reason: length` endet, wirft einen
  erklärenden `ProviderError` statt einer leeren Sprechblase.

### Phase 3A — Voice-Schicht: Provider, Policy, Chunker

`kiki/voice/tts/` ist der neutrale Unterbau der Sprachausgabe. Er ist frei von
torch, CUDA, GTK und Audiogeräten — die Oberfläche importiert ihn, und nichts
darin darf als Nebenwirkung eine GPU belegen. Ein Test prüft das, indem er das
Paket in einem eigenen Prozess importiert und `sys.modules` inspiziert.

**Warum weder FlashAttention noch `torch.compile`.** Gemessen an der aktuellen
Qwen3-TTS-0.6B-Inferenz, 4,6 s Audio:

| | |
|---|---|
| Gesamtdauer Synthese | 6,75 s (RTF ≈ 1,33) |
| `scaled_dot_product_attention` | 0,169 s = **2,5 %** |
| `torch.nn.linear` | 0,928 s bei 43.013 Aufrufen |
| Modul-Aufrufe gesamt | 97.593 |

FlashAttention beschleunigt die 2,5 %. Selbst bei perfekter Beschleunigung bliebe
der Gewinn unter drei Prozent. `torch.compile` wurde gemessen: **1 %**. Die
GPU ist Blackwell `sm_120`, für das es kein sicheres Wheel gibt, und KIKI setzt
`attn_implementation="sdpa"` fest, sodass ein Build ohne weiteren Umbau gar nicht
benutzt würde. Der Engpass ist autoregressives Audio-Token-Decoding mit sehr
vielen kleinen Aufrufen, nicht Attention.

**Aufbau:**

| Modul | Aufgabe |
|---|---|
| `models.py` | `TTSRequest`, `AudioChunk`, `TTSHealth`, Status, Capabilities, Ergebnis |
| `provider.py` | `TTSProvider`-Protocol: `load`/`unload`/`synthesize`/`cancel` |
| `policy.py` | Voice Response Policy: Modus, Längenkappung, Redaktion |
| `answer.py` | Plant eine vollständige Mikrofonantwort einmalig für kompaktes TTS und entscheidet, ob der Volltext-Chat geöffnet wird |
| `chunker.py` | deutscher Streaming-Chunker |
| `fake.py` | `FakeTTSProvider` und `NullTTSProvider` für Tests |

**Policy.** Redaktion läuft vor der Längenkappung: Erst den Codeblock entfernen
heißt, das Satzbudget wird für Prosa ausgegeben statt für einen Zaun, der ohnehin
übersprungen worden wäre. Secrets werden zuerst entfernt, damit ein Token in
einem Codeblock nicht überlebt, weil eine spätere Regel den Block behielt.
`SpeechPlan.removed` nennt nur Kategorien, nie Inhalte — ein Diagnosefeld darf
nicht der Weg sein, auf dem ein Schlüssel in ein Log gelangt.

**Chunker.** Entscheidend sind die negativen Regeln: Ein Punkt beendet keinen
Satz in „z.B.", „1,33", „Fedora 44.1", „192.168.0.1", einer URL, einem Pfad oder
einem Markdown-Link. Der erste Chunk wird früh geschnitten, weil die Stille vor
dem ersten Ton praktisch seine Länge ist; spätere Chunks bleiben länger, damit
die Betonung natürlich bleibt.

### Signale der Sprachausgabe: `on_speaking`, `on_audio_started`, `on_idle`

Der `SpeechDirector` meldet drei Dinge, und sie bedeuten nicht dasselbe:

| Signal | Bedeutung |
|---|---|
| `on_speaking` | Der Sprachauftrag wurde angenommen. **Nicht** „es ist etwas zu hören." |
| `on_audio_started` | Der erste Chunk mit echten Samples geht an die Wiedergabe. Ab hier ist KIKI hörbar. |
| `on_idle` | Die Äußerung ist beendet, abgebrochen oder fehlgeschlagen. |

Die Character-State-Machine wechselt bei `on_audio_started` von `thinking` auf
`speaking`. `on_speaking` stummschaltet nur das Mikrofon — früh ist dort
harmlos, spät ließe KIKI sich selbst hören.

**Warum das nötig wurde.** Auf der Controller-Route liegen zwischen Annahme und
erstem Ton mehrere Sekunden: der Dienst antwortet mit einem *vollständigen* WAV,
also existiert vor dem Ende der Synthese kein einziges Sample. Gemessen gegen den
laufenden Dienst: `time_to_first_audio` = 5,32 s bei 2,88 s Audio. Hätte die
Animation weiter an `on_speaking` gehangen, hätte KIKI die halbe Zeit stumm
geredet.

**Beide Routen senden das Signal.** Auf der Dateiroute fällt es mit
`on_speaking` zusammen — das WAV ist fertig, wenn es den Player erreicht, „an-
genommen" und „hörbar" sind dort derselbe Augenblick. Dadurch braucht der
Zuhörer keine Fallunterscheidung nach Route.

**Wo genau es ausgelöst wird.** Im `VoicePlaybackController`, für den ersten
Chunk mit Samples, der die Prüfungen besteht (richtige Request-ID, nicht
abgebrochen, nicht verworfen). Der Aufruf an den Sink wird dafür zu einem Task
und bekommt eine Loop-Runde: das reicht `PipeWireAudioSink`, um seine Datei zu
schreiben und die Pipeline zu starten — beides passiert vor seinem ersten
`await`. Scheitert er in dieser Strecke, wurde nie ein Ton erzeugt und nichts
gemeldet. Vorher zu melden würde Audio behaupten, das ein defekter Sink nie
erzeugt; nachher zu melden käme erst, wenn der Chunk schon vorbei ist.

**Thread-Grenze.** Der Controller ruft auf dem asyncio-Thread zurück.
`kiki.voice.tts` importiert kein GLib; die Anwendung reicht das Ereignis mit
`GLib.idle_add` an den GTK-Main-Thread weiter — derselbe Weg, den Weckwort und
Watcher schon nehmen. Der `SpeechDirector` verwirft dabei jedes Ereignis, dessen
Request-ID nicht zur laufenden Äußerung gehört: eine Antwort, die der Nutzer
unterbrochen hat, kann sich hinterher nicht mehr melden.

**Was das Signal nicht ist.** Es markiert den Beginn der *lokalen Wiedergabe*,
nicht den Beginn der Modell-Synthese. Solange der Dienstpfad ganze WAV-Dateien
liefert, kann „Audio gestartet" erst nach dem Ende der Synthese eintreten. Echtes
PCM-Streaming — der Dienst schickt Chunks, während er noch rechnet — bleibt ein
eigener, späterer Optimierungsslice; erst der verkürzt den Abstand zwischen
`on_speaking` und `on_audio_started`.

### Phase 2H — Eigener LLM-Harness

`services/kiki-llm/` ist KIKIs eigene Modell-Laufzeit auf PyTorch/transformers —
weder Ollama noch llama.cpp. Sie läuft als eigener Dienst, aus demselben Grund
wie der TTS-Dienst: PyTorch und CUDA kommen dem GTK-Prozess nicht nahe, ein
CUDA-Fehler beendet eine systemd-Unit statt des Desktop-Pets.

Was der eigene Harness kann, was ein fremder Server nicht hergibt:

| | |
|---|---|
| Denken | `bad_words_ids` auf die `<think>`-Token-IDs — das Modell *kann* keinen Denkblock öffnen |
| Werkzeuge | Deklarationen über Qwen3s eigenes Chat-Template, Aufrufe aus dem Stream geparst |
| Slots | Zulassung nach Priorität; ein Hintergrundjob blockiert kein Gespräch |
| VRAM | eigenes Budget, teilt die Karte mit dem TTS-Dienst statt darum zu kämpfen |

**Protokoll** (NDJSON, Loopback):

```
{"delta": "Text"}
{"tool_call": {"id": …, "name": …, "arguments": {…}, "parse_error": ""}}
{"done": true}
```

**Der Stream-Parser ist der heikle Teil.** `<tool_call>` kommt regelmäßig über
zwei Chunks verteilt an (`"<tool"` + `"_call>"`). Text, der noch ein Tag werden
könnte, wird deshalb zurückgehalten, bis es entschieden ist — ein optimistischer
Parser schreibt halbe Tags ins Chatfenster. Ein abgebrochener Aufruf wird als
Fehler gemeldet, nie geraten.

**VRAM-Budget.** Prioritäten aus dem Kontextplaner: `exclusive` verdrängt `high`,
`high` verdrängt `low`, `low` verdrängt niemanden, gleiche Stufe fasst Nachbarn
nicht an. Angepinnte Residents bleiben unantastbar, verdrängt wird nur so viel
wie nötig, und ein Headroom bleibt frei, damit der Compositor nicht ruckelt.

**Was lokal ist, ist das Modell — nicht die Werkzeuge.** Ein Tool-Handler darf
weiterhin einen Cloud-Dienst aufrufen; die Registry und die Policy ändern sich
dadurch nicht. Für Cloud-*Modelle* bleibt `ai.provider = "openai_compatible"`.

**Echtes Batching** (`batching.py` + `torch_batch.py`): Ein Forward-Pass bedient
alle aktiven Sequenzen. Der Scheduler ist tensorfrei und entscheidet nur, wer
beitritt, wer gemeinsam dekodiert und wer ausscheidet — dadurch sind die
unangenehmen Fälle ohne GPU testbar (Sequenz endet mitten im Batch, Anfrage
kommt während des Dekodierens, Client bricht ab, Prefill wirft).

Zwei Dinge, die dabei zählen:

- **Linkspadding.** Generierung liest immer die letzte Position, also werden
  unterschiedlich lange Sequenzen am rechten Rand ausgerichtet und das Padding
  vorne maskiert. Rechtspadding ließe das Modell Padding als neuesten Input lesen.
- **Ein persistenter Batch-Cache.** Chirurgie nur beim *Wechsel* der Besetzung:
  Beitritt ist Padden und Anhängen, Austritt ein `index_select`. Ein erster
  Entwurf zerlegte und verschmolz den Cache bei jedem Schritt — das ist O(Cache)
  an Speicherverkehr pro Token und wäre langsamer gewesen als gar nicht zu batchen.

Ein **Batch-Fenster** von 20 ms beim Formieren eines leeren Batches: Ohne das
nimmt die Schleife die erste Anfrage an und dekodiert los, bevor die zweite
eingereicht ist — es entstünde nie ein Batch. Eine einzelne Anfrage zahlt das
Fenster einmal, gegen eine Generierung in Sekunden.

`exclusive` (Code-Review) wartet, bis der Batch leer ist, statt einen Pass zu
teilen — dafür ist die Stufe da.

### Phase 2G — Dynamisches Kontextbudget

`kiki/context/` bemisst den Kontext pro Zug, statt ein festes Fenster zu
reservieren. Gemessen auf RTX 5060 Ti mit qwen3-vl:4b, jede Messung mit
einzigartigem Text (sonst greift Ollamas Prompt-Cache und alles wirkt gratis):

| Prompt-Tokens | Prefill | 1. Token |
|---|---|---|
| 61 | 0,02 s | 0,44 s |
| 3.009 | 0,59 s | 1,10 s |
| 8.587 | 1,95 s | 2,75 s |
| 23.652 | 8,94 s | 10,41 s |
| 32.758 | 15,92 s | 28,88 s |

Rund 4.400 Token/s Prefill: jede weiteren 1.000 Tokens kosten etwa eine
Viertelsekunde, bevor KIKI ein Wort sagt.

Budgets je Anliegen (`Intent`), Fenster jeweils auf die nächste Zweierpotenz mit
Reserve für die Antwort:

| Intent | Budget | num_ctx | Priorität |
|---|---|---|---|
| `smalltalk` | 2.000 | 4.096 | hoch |
| `advice` | 6.000 | 8.192 | hoch |
| `coding_plan` | 12.000 | 16.384 | hoch |
| `coding_review` | 16.000 | 32.768 | exklusiv |
| `background` | 2.000 | 4.096 | niedrig |

Füllreihenfolge ist L1 (Turn und Regeln), dann **L3** (Erinnerungen), dann L2
(Verlauf) — bewusst so: Ein paar relevante Fakten sind mehr wert als vier weitere
Zeilen Small Talk, also wird der Verlauf gekürzt, nicht das Gedächtnis.

Klassifikation und Retrieval kosten **null Modell-Tokens** — Schlüsselwort-
Heuristik plus SQLite. Die Token-Schätzung ist gegen Ollamas eigenen
`prompt_eval_count` kalibriert (2,83–3,95 Zeichen/Token für Deutsch, Code und
Gemischtes) und rechnet bewusst mit 3,0, um eher zu über- als zu unterschätzen.

### Phase 2F — Öffnen ohne Nachfrage

Die sieben deklarierten PC-Aktionen hingen bis 2F ausschließlich am
Kontrollfenster: `workspace_tools.py` erklärte sie, aber mit `handler=_EMPTY`
und ohne Registrierung. Das Modell hatte gar keinen Weg dorthin.

`kiki/tools/launch_tools.py` leitet daraus **ausführbare Kopien** ab
(`dataclasses.replace`) — die Deklarationen für die Kontrollfenster-Vorschau
bleiben unverändert. Dazu eine neue Risikostufe:

`RiskLevel.LAUNCH` — öffnet etwas Sichtbares auf dem eigenen Desktop, ändert
keine Daten, und was aufgeht, treibt weiterhin der Nutzer. Bewusst getrennt von
`WRITE`, damit „Projektordner öffnen" nicht als Datenänderung eingestuft werden
muss, um erlaubt zu sein.

| Stufe | Unbeaufsichtigt |
|---|---|
| `strict` | READ |
| `balanced` (Default) | READ, CONTROL |
| `trusted` | READ, CONTROL, LAUNCH |

WRITE und EXTERNAL fehlen in **jeder** Stufe. Zwischenablage und
Benachrichtigung bleiben deshalb draußen: Die Zwischenablage zu ersetzen ändert
Daten des Nutzers.

Was die Lockerung eingrenzt:

- Jeder Pfad kommt aus `WorkspaceRegistry.require()`, das Roots, Symlinks und
  Git-Wurzel **bei jedem Aufruf** neu prüft. Ein nach der Registrierung
  umgebogener Pfad wird abgelehnt, nicht weiterbenutzt.
- Dateien müssen über `resolve_inside_workspace` im Workspace bleiben.
- Terminal und Editor nutzen feste Argv-Vorlagen aus einer Allowlist. Kein
  Shell-String, kein `-c` — Modelltext wird nie zum Kommando.
- URLs nur `http`/`https`, ohne Zugangsdaten.
- Der Nutzerpfad (`Origin.USER`) ist unverändert: LAUNCH verlangt dort weiterhin
  eine Bestätigung, damit das Kontrollfenster seine Freigabekarte behält.
- Panic entfernt die Werkzeuge auf jeder Stufe.

Die Obergrenzen des Agent-Loops (`max_tool_calls`, Wiederholungserkennung)
begrenzen, wie viel ein einzelner Zug öffnen kann. Eine zugübergreifende
Startbremse gibt es noch nicht.

### Phase 2E — Proaktivität

`kiki/watch/` kehrt um, dass KIKI ausschließlich auf Ansprache reagiert. Der
Aufbau ist dreiteilig und bewusst getrennt:

```
Watcher (beobachtet)  →  Notice (Text)  →  Notifier (entscheidet)  →  Ausgabe
```

**Die Sicherheitsgrenze steckt im Datentyp.** Ein `Notice` trägt Text und sonst
nichts — kein Callback, kein Tool, keine Parameter. KIKI gewinnt damit die
Fähigkeit, sich von selbst zu **melden**, aber ausdrücklich nicht die, von selbst
zu **handeln**. Jede Systemänderung behält ihre Freigabekarte. Das ist der
Unterschied zu „Autonome Automationen", die weiter ausgeschlossen bleiben.

**Watcher sind flankengesteuert.** Ein Akku bei 19 % meldet sich einmal, nicht
bei jedem Poll. Die Bedingung muss sich erst wieder auflösen (Ladekabel, Wert
über Schwelle plus Marge), bevor erneut gemeldet wird. Ein weiteres Absacken
meldet sich erneut. Ohne das müsste der Cooldown im Notifier Arbeit leisten, die
der Watcher selbst erledigen sollte.

**Der Notifier ist reine Entscheidungslogik** mit injizierter Uhr, damit die
unangenehmen Fälle testbar bleiben:

| Situation | Benachrichtigung | Sprache |
|---|---|---|
| Warnung, tagsüber | ja | ja |
| Warnung, Ruhezeit | nein | nein |
| Dringend, Ruhezeit | ja | nein |
| Info, jederzeit | ja | nie |
| Nutzer spricht gerade mit KIKI | ja | nein |
| Panic | nein | nein |

Dazu ein Cooldown je `key` (Default 30 min) und ein Stundenbudget (Default 6),
das ein durchgedrehter Watcher nicht überschreiten kann. Eine unterdrückte
Meldung verbraucht kein Budget.

Der `WatchService` pollt auf dem asyncio-Thread und führt jeden `check()` in
einem Worker-Thread aus (D-Bus und `statvfs` blockieren). Ein Watcher, der wirft,
wird protokolliert und übersprungen — er darf weder die anderen noch die Schleife
stoppen.

Vorhandene Watcher: `battery` (Warnung unter 20 %, dringend unter 10 %, nur im
Akkubetrieb) und `disk` (Warnung ab 90 %, dringend ab 96 %). Kalender- und
Build-Watcher fehlen noch; der Kalender bräuchte eine neue
`evolution-data-server`-Anbindung.

### Phase 3A — Jarvis-Modus und Desktop-Steuerung

`AutonomyLevel.JARVIS` erweitert die Tabelle unbeaufsichtigter Risikostufen um
WRITE und EXTERNAL — die einzige Stelle, an der beide vorkommen. Es ist eine
bewusste Nutzerentscheidung (Einstellungen → Vertrauensstufe, nie Default),
kein Zustand, in den die Anwendung von selbst gerät. Was auch in dieser Stufe
weiterhin stoppt: die Hard-Deny-Liste, unbekannte Tools, der Panic-Schalter,
der Integrationsschalter, das `model_callable`-Gate — und jedes Tool, dessen
Autor `auto_allow` nicht gesetzt hat. Dieses Autoren-Veto rangiert über jeder
Stufe; `routines.create`, `routines.delete`, Zwischenablage und Gedächtnis
behalten ihre Karte daher selbst im Jarvis-Modus.

Der `Origin.USER`-Pfad ist unverändert: Ein Klick fragt so oft wie vorher.
Der Modus weitet, was das *Modell* entscheiden darf, nicht was ein Dialog
überspringt.

Fünf neue Skills erreichen den Desktop über dieselbe Werkzeug-Schicht wie
alles andere (ToolSpec → Policy → Executor → Audit):

| Skill | Werkzeuge | Risiko | Weg |
|---|---|---|---|
| `media_control` | `media.status`, `media.play_pause/next/previous/stop` | READ/CONTROL | MPRIS über den Sitzungsbus; `Can*`-Antworten des Players werden respektiert |
| `audio_control` | `audio.volume_get/set`, `audio.mute` | READ/CONTROL | `pactl` mit festem argv, numerisch validiert, `LC_ALL=C` gegen Locale-Fallen |
| `display_control` | `display.brightness_get/set` | READ/CONTROL | GNOME-SettingsDaemon first, Plasma-6.3+-Displays als Fallback (kein gemeinsames API beider Desktops) |
| `app_launch` | `app.list`, `app.open` | READ/LAUNCH | Index aus den zwei XDG-Verzeichnissen; Start nur über `gio launch <datei>`, kein Exec-Parsen |
| `session_control` | `session.lock` | CONTROL | `org.freedesktop.ScreenSaver`, Fallback `org.gnome.ScreenSaver` |

D-Bus-Zugriff liegt in `kiki/platform/dbus.py` hinter kleinen Klassen, die
Tests ersetzen können; kein Test braucht einen echten Bus. Suspend, Reboot,
Netzwerk und Pakete sind bewusst nicht dabei — sie unterbrechen KIKI selbst
oder stehen auf der Hard-Deny-Liste und bleiben ein eigener Entschluss.

### Phase 3B — Freigegebene Routinen

Routinen sind die einzige Form, in der KIKI ohne Ansprache **handelt**. Ein
Rezept paart eine Messgröße (`battery.percent`, `disk.used_percent`) mit einem
gespeicherten Werkzeugaufruf:

```
Routine = Trigger (Metrik, Vergleich, Schwelle) + Tool-Name + Argumente + Cooldown
```

Der entscheidende Unterschied zum Watcher: Ein `Notice` bleibt Text, eine
Routine darf handeln — aber nur das exakt bestätigte Rezept. `routines.create`
zeigt die Karte mit Auslöser, Werkzeug, Argumenten und Abklingzeit wortwörtlich
an; `auto_allow=False` sorgt dafür, dass diese Karte auf **jeder** Stufe
erscheint, Jarvis eingeschlossen. Verwaltung über `routines.list/toggle/delete`
oder die Einstellungsseite.

Die Ausführung läuft durch denselben `ToolExecutor` wie alles andere, mit
`Origin.ROUTINE`:

- Policy: erlaubt nur Tools mit `auto_allow=True` (Autoren-Veto gilt). Eine
  dauerhafte Verweigerung (Werkzeug weg, Freigabe entzogen) schaltet die
  Routine ab, statt sie in jedem Tick erneut ins Audit zu schreiben.
- Panic und Integrationsschalter werden bei jedem Tick neu gelesen.
- Cooldown (Default 30 min) verhindert, dass eine festsitzende Messgröße ein
  Werkzeug hämmert; `battery.percent` existiert nur im entladenden Zustand,
  damit eine ladende Batterie keine Tiefstand-Routinen auslöst.
- Das Audit trägt `Origin.ROUTINE`; die Messgrößen kommen aus denselben
  Integrationen, die die Watcher lesen — keine zweite D-Bus-Fläche, kein Cache.

### Phase 3C — System & Netzwerk

Zwei weitere Skills erreichen die Maschine über den **Systembus** — die erste
Stelle, an der KIKI über den eigenen Desktop hinausgreift:

| Skill | Werkzeuge | Risiko | Weg |
|---|---|---|---|
| `network_control` | `network.wifi_list`, `network.wifi_set`, `network.vpn_list`, `network.vpn_connect`, `network.vpn_disconnect` | READ/CONTROL | NetworkManager D-Bus: Funkgerät über die Property `WirelessEnabled`, Verbindungen über `ActivateConnection`/`DeactivateConnection` per UUID |
| `power_control` | `power.suspend`, `power.reboot`, `power.poweroff` | CONTROL/WRITE | logind `Suspend`/`Reboot`/`PowerOff`; polkit erlaubt der aktiven Sitzung alle drei ohne Passwort |

Die Risikotrennung trägt die Abwägung: **Suspend** ist CONTROL — die Maschine
schläft und wacht auf, es geht nichts verloren, „Schlafmodus“ soll einfach
funktionieren. **Reboot und PowerOff** beenden alles Ungespeicherte, KIKI
eingeschlossen, und sind WRITE: außerhalb des Jarvis-Modus immer Karte, im
Jarvis-Modus genau der Handel, den die Stufe beschreibt. Login/Logout fehlen
bewusst — sie beenden den KIKI-Prozess, bevor er Erfolg melden könnte.

**SSID-Schutz gilt weiter — mit einer neuen Grenze.** Die passive Integration
(`integrations/networkmanager.py`) meldet nach wie vor keine SSID: Sie läuft
ungefragt auf jeder Statuskarte. Die Werkzeuge dagegen laufen, *weil der Nutzer
nach Netzwerken gefragt hat* — da ist das Nennen der SSID die Antwort, nicht
ein Leak. Die Grenze liegt damit zwischen „unggefragt ständig sichtbar“ und
„auf Abruf“.

**Verbindungen verbinden, nicht verändern.** Das Werkzeug aktiviert und
deaktiviert vorhandene Verbindungen per UUID; die UUID kommt aus
`network.vpn_list`. Verbindungen anlegen, bearbeiten oder Passwörter übergeben
kann es nicht — das bleibt Systemeinstellungs-Territorium. GetSettings liefert
ohnehin keine Secrets.

`install_package` bleibt auf der Hard-Deny-Liste. Paketverwaltung ändert die
Maschine grundlegend und braucht eine eigene Entschärfungsentscheidung mit
eigenen Guardrails — nicht einfach ein weiteres Tool.

### Sicherheitsinvarianten seit 0.5.0

- Eine Umsetzung akzeptiert nur eine erfolgreich beendete Plan-Session mit gleichem Workspace und unverändertem Aufgabentext; ihre `plan_session_id` bleibt an der Umsetzung gespeichert.
- Stop verändert ausschließlich eine bekannte, noch laufende Session. Pro Workspace darf zugleich nur eine aktive Agent-Session existieren.
- Ein Testlauf mit Session-Bezug muss demselben Workspace angehören. Stdout und stderr werden parallel geleert; UI und Persistenz erhalten eine auf 64 KiB begrenzte, bei Bedarf markierte Ausgabe.
- TTS-Stop löscht Warteschlangen, bricht den aktiven Syntheseauftrag ab und verwirft Ergebnisse älterer Synthese-Generationen.
- PC-Aktionen laufen ausschließlich über eine feste Siebener-Allowlist im Observe-Profil. Gesprochener Text darf nur das Kontrollfenster öffnen, nie eine Aktion ausführen.
- Zwischenablage- und Benachrichtigungstexte erscheinen in der sichtbaren Vorschau, werden aber in persistenten Freigabeparametern redigiert und nicht im Audit wiederholt.
- Die Mikrofon-Pipeline wartet begrenzt auf EOS, bevor sie auf NULL wechselt, damit `wavenc` einen gültigen RIFF/WAV-Header schreibt.

### Nur als Schnittstelle vorbereitet

Die OpenCode-Anbindung hat weiterhin zwei harte Grenzen: `observe` ist heute
ein Prompt-/Policy-Vertrag mit nachgelagerter Worktree-Prüfung, keine
Betriebssystem-Sandbox oder Netzwerksperre. Außerdem bindet die sichtbare
Freigabe die gesamte Umsetzungs-Session; OpenCode-interne Tool-Anfragen werden
protokolliert, aber noch nicht einzeln angehalten und von KIKI freigegeben.

- Skill-Registry / Plugin-Marktplatz
- Automatisch abgeleitete Erinnerungen (heute nur ausdrücklich bestätigte)
- Weitere Watcher (Kalender, Build-/Testläufe, Homelab)
- Modell-Tool-Use jenseits der `status_*`-Tools (weitere Tools brauchen je ein `model_callable`)
- Bestätigungspflichtige Write-Tools (Dialog existiert, Capture-Screen braucht UI)
- Globale Hotkeys jenseits von In-App-Accelerators, Notifications, RAG
- Blockierender Freigabe-Handshake für OpenCode-interne Tool-Calls
- Echte Podman-/Distrobox-Isolation (Stubs deny)
- Homelab-Skills (Proxmox, UniFi, …)
- Renderer lottie / spine / live2d / godot
- Layer-shell-Backend

### Bewusst nicht im MVP

- Beliebige Shell aus Modelltext
- Automatisches Anreichern von Prompts mit Systemdaten — **eine** Ausnahme seit
  Phase 2C: der Gedächtnisblock, und darin steht nur vom Nutzer Freigegebenes
- Root-Aktionen
- Freie autonome Automationen. Seit Phase 2E darf KIKI sich von selbst
  **melden**; seit Phase 3B darf sie exakt die Wenn-Dann-Rezepte ausführen,
  die der Nutzer als Routine bestätigt hat. Alles andere tut sie weiterhin
  nur auf Ansprache.

## 5. Charakter-Asset-Vertrag

`data/character/<id>/manifest.toml` beschreibt Clips. Die App kennt nur `CharacterPack` + `AnimationEngine`. Ein späterer Live2D-Renderer implementiert dieselbe Clip-API.

Die mitgelieferte Figur ist eine **eigens erzeugte** 2D-Illustration (kein Fedora-Logo, keine Fremdmarke). Outfit in Blau/Weiß/Silber/Violett.

## 6. Sicherheitsfluss

```
Anfrage (origin = user | model) → Registry.get(name)
       → HARD_DENY? → Audit deny → Stop
       → unbekannt? → Audit deny → Stop
       → FORBIDDEN? → Audit deny → Stop
       → Panic? → Audit deny → Stop
       → Integrationen aus? → Audit deny → Stop
       → Profil verboten? → Audit deny → Stop
       → origin=model und nicht model_callable? → Audit deny → Stop
       → validate_params (kein additionalProperties)
       → auto_allow = false → Vorschau-Dialog
       → origin=model: Risiko in Vertrauensstufe? → ausführen : Vorschau-Dialog
       → origin=user:  READ/CONTROL → ausführen : Vorschau-Dialog
       → Dialog → Confirm/Cancel → Audit → ggf. ausführen
```

Jeder Audit-Eintrag trägt seinen Ursprung (`[user]` / `[model]`), sodass im
Nachhinein unterscheidbar bleibt, was ein Klick und was eine Modellentscheidung
war.

`origin = user` verhält sich exakt wie vor Phase 2A. Der Agent-Loop
(`kiki/ai/agent_loop.py`) benutzt ausschließlich `origin = model` und erreicht
damit nur Tools, die ausdrücklich `model_callable = true` gesetzt haben —
derzeit die vier `status_*`-Tools. Der Loop ist auf Schritte, Aufrufanzahl,
Wiederholungen und Ergebnisgröße begrenzt; die Tool-Liste wird pro Zug neu aus
der Policy erzeugt, damit der Panic-Schalter mitten im Zug greift.
