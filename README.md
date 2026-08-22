# KIKI

Freundliches 2D-KI-Desktop-Pet für **Fedora Linux 44** (GNOME, Wayland).

KIKI sitzt als kleine Figur auf dem Desktop, öffnet auf Klick einen Chat und spricht mit einem lokalen Ollama-Modell. Sie führt **keine** Systemänderungen aus, nur weil das Modell Text erzeugt hat.

Architektur, Wayland-Grenzen und der MVP-Schnitt: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Als Fedora-App installieren

KIKI wird als natives Fedora-44-`x86_64`-RPM gebaut. Das Paket installiert den
`kiki`-Befehl, die Charakterbilder, GNOME-Desktopdatei, AppStream-Metadaten,
Icons und eine optionale systemd-User-Unit gemeinsam.

Der vollständige Ein-Datei-Installer ist der einfachste Weg. Er enthält das
KIKI-RPM, prüft Fedora-Version, Architektur, RPM-Header und SHA-256, lässt DNF
alle Fedora-Abhängigkeiten auflösen und bietet danach die vorbereitete Vosk-
Spracherkennung, Ollama samt lokalem Modell sowie die für Coding-Sessions
benötigte OpenCode-Terminal-CLI jeweils kontrolliert an:

```bash
chmod +x KIKI-0.8.0-Fedora-44-x86_64.run
./KIKI-0.8.0-Fedora-44-x86_64.run
```

Das RPM ist eingebettet; Fedora-Pakete und Modelle werden aus den
konfigurierten Paketquellen bzw. ihren offiziellen HTTPS-Quellen geladen. Der
Installer ist daher eine einzige Startdatei, aber kein vollständig
netzunabhängiges Abbild. Ollama ist von der schlanken Grundinstallation
getrennt, da das Fedora-Paket je nach GPU-Stack mehrere GB belegen kann.
Qwen3-TTS ist hardwareabhängig und wird nur mit `--with-gpu-tts` eingerichtet.
Optionen zeigt `--help`.

Installer und RPM selbst bauen:

```bash
sudo dnf install -y \
  rpm-build appstream desktop-file-utils systemd-rpm-macros \
  gdk-pixbuf2 git-core python3 python3-cairo python3-cffi \
  python3-gobject python3-httpx python3-pillow python3-pytest \
  espeak-ng

./scripts/build-rpm.sh
./scripts/smoke-test-rpm.sh
./scripts/build-fedora-installer.sh
./scripts/smoke-test-installer.sh
sudo dnf install ./dist/kiki-0.8.0-1.fc44.x86_64.rpm
```

Die lokal erzeugten Entwicklungsartefakte sind noch nicht mit einem
Release-GPG-Schlüssel signiert. Vor einer öffentlichen Verteilung müssen RPM
und Installer aus einer sauberen Fedora-44-Buildumgebung signiert werden.

Danach erscheint **KIKI** in der GNOME-Appübersicht; alternativ startet
`kiki` die App. Die Einstellung „Bei Anmeldung starten“ verwaltet den
XDG-Autostart unter `~/.config/autostart/`.

Ein Upgrade verwendet denselben `dnf install`-Befehl mit dem neuen RPM. Zum
Entfernen:

```bash
sudo dnf remove kiki
```

Persönliche Konfiguration, SQLite-Daten, Logs und lokal geladene Modelle in
den XDG-Benutzerverzeichnissen bleiben bei der Deinstallation absichtlich
erhalten.

## Lokales KI-Modell und optionale Sprache

Ollama ist der lokale Default. KIKI erwartet ein Modell mit Deutsch und Vision:

```bash
sudo dnf install ollama
sudo systemctl enable --now ollama
kiki-setup-model      # nach RPM-Installation
# im Quellbaum alternativ: ./scripts/setup-local-model.sh
# gleichbedeutend: ollama pull qwen3-vl:4b
```

| Modell | Größe | Wofür |
|---|---|---|
| **`qwen3-vl:4b`** (Default) | ~3,3 GB | Deutsch, Chat, Screenshots, OCR |
| **`qwen3-vl:8b`** (Qualitätsprofil) | ~6 GB | Natürlichere Persona, bessere Planung und Coding-Hilfe |
| `qwen3-vl:2b` | ~1,9 GB | Schwächer, weniger RAM |
| `gemma3:4b` | ~3,3 GB | Sehr gutes Mehrsprachen-Chat + Vision |

Der 4B-Default bleibt bewusst erhalten: Er startet auf mehr Rechnern und ein
Update zieht dadurch kein größeres Modell nach. Auf einem System mit genügend
RAM oder VRAM ist 8B das empfohlene Qualitätsprofil:

```bash
kiki-setup-model qwen3-vl:8b
# danach in Einstellungen → Ollama-Modell: qwen3-vl:8b
```

KIKIs Systemrolle ist auf direkte deutsche Antworten, ehrliche Werkzeuggrenzen,
den getrennten Coding-Agenten und kurze, gut vorlesbare Sätze abgestimmt. Der
Ton ist über Personas umschaltbar, die festen Regeln bleiben davon unberührt —
beides ist in den Einstellungen einsehbar.

Bilder gehen nur ins Modell, wenn du eine Datei anhängst **oder** ausdrücklich „Bildschirm zeigen“ freigibst. KIKI fotografiert den Desktop **nicht** von allein.

Spracheingabe ist Push-to-talk und vollständig lokal mit Vosk; optional gibt es
zusätzlich das Weckwort **„KIKI"** (siehe unten, Default aus). Das RPM bindet
Fedoras gepflegte Vosk-0.3.50-Laufzeit ein; eine separate Pip-Installation ist
nicht nötig. Laufzeit und OpenFST-Abhängigkeit benötigen zusammen etwa 9 MB
Download beziehungsweise 63 MB installiert. Das deutsche Modell (~45 MB)
wird beim ersten Zuhören nach
`~/.local/share/kiki/vosk/vosk-model-small-de-0.15` geladen.
Der Komplettinstaller lädt es vorab. Manuell geht derselbe abgesicherte,
größenbegrenzte, SHA-256-gepinnte und atomare Setup-Schritt mit
`kiki --prepare-voice-model`.

Sprachausgabe nutzt bevorzugt einen **eigenen GPU-Dienst** (nicht im
GTK-Prozess). Solange dieser noch nicht eingerichtet oder vorübergehend nicht
erreichbar ist, spricht KIKI automatisch über die lokale Fedora-Systemstimme
(`espeak-ng`). Stop/Barge-in bricht auch eine noch laufende Synthese ab;
verspätet fertiggestellte Audiodateien werden verworfen.

```
KIKI GTK-App
    ├── Ollama ── qwen3-vl:4b ── Chat-Antworten
    ├── TTS-Service ── Qwen3-TTS 0.6B CustomVoice ── CUDA
    └── PipeWire ── Audio-Ausgabe
```

```bash
# voller Stack (Python 3.12 + CUDA + qwen-tts, ~1–2 GB Modell)
sudo dnf install -y python3.12 python3.12-devel
kiki-setup-tts        # nach RPM-Installation
# im Quellbaum alternativ: ./scripts/setup-tts.sh

# nur die Leitung testen (kurzer Ton, kein GPU-Modell):
python3 services/qwen3-tts/kiki_tts_server.py --dummy
```

Default-Stimme: **Serena** (warm, jung, weiblich), Sprache **German**. Der Dienst lauscht nur auf `127.0.0.1:18765`.

### Warum ein eigener Dienst und keine Integration in die App

Gemessen auf einem Rechner mit RTX 5060 Ti:

| | Transport (HTTP über Loopback) | Modellrechnung |
|---|---|---|
| Sprachausgabe | 0,48 ms | 1.815 ms (**99,97 %**) |
| Ollama, bis zum ersten Token | 2,79 ms | 15.194 ms (**99,98 %**) |

Die Schnittstelle kostet also weniger als ein Promille. Alles in den GTK-Prozess
zu holen brächte keine spürbare Geschwindigkeit, würde aber drei Dinge kaputt
machen: PyTorch im Oberflächenprozess friert die Oberfläche während jeder
Inferenz ein, ein CUDA-Fehler risse die ganze App mit statt nur den Dienst (der
sich per `Restart=on-failure` selbst fängt), und Ollama ist ein Go-Programm, das
sich gar nicht in Python laden lässt.

Dieselbe Trennung ist das, was den Wechsel in die Cloud überhaupt erlaubt:
`ai.provider = "openai_compatible"` und `tts.base_url` zeigen auf einen
beliebigen Endpunkt. Für **STT gibt es diese Naht noch nicht** — die
Spracherkennung ist fest an lokales Vosk gebunden.

## Entwicklung starten

```bash
sudo dnf install -y \
  python3-pip python3-gobject python3-cairo gtk4 libadwaita libsecret \
  python3-httpx python3-pillow python3-pytest
git clone <repo> ProjectKIKI
cd ProjectKIKI
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m kiki --debug
```

`--system-site-packages` ist nötig, damit PyGObject, GTK4 und libadwaita aus Fedora sichtbar bleiben.

Ohne venv:

```bash
PYTHONPATH=src python3 -m kiki --debug
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest
```

Die Unit-Tests brauchen **kein** Display und kein Ollama.

## Erste Schritte

1. Komplettinstaller ausführen; alternativ Ollama starten und `qwen3-vl:4b` ziehen (`kiki-setup-model`).
2. `kiki` starten. Die Figur erscheint als kleines Fenster.
3. **Ziehen** zum Verschieben, **Linksklick** öffnet den Chat, **Rechtsklick** das Menü.
4. In den Einstellungen URL und Modell prüfen → „Verbindung testen“.
5. Optional: Provider auf OpenAI-kompatibel stellen. Empfohlen: SpaceXAI mit `https://api.x.ai/v1` und Modell `grok-4.5`. Der API-Key landet im GNOME Keyring.

## Bedienung

| Aktion | Effekt |
|---|---|
| Linksklick auf KIKI | Chat öffnen |
| Ziehen | Fenster verschieben (Wayland: interaktiv über den Compositor) |
| Rechtsklick | Chat, Pause, Einstellungen, Neu laden, Fenstermenü, Beenden |
| Enter im Chat | Senden |
| Shift+Enter | Neue Zeile |
| Bild-Button / Drag-and-drop | Hängt eine Bilddatei an die nächste Nachricht. |
| Bildschirm-Button / Rechtsklick „Bildschirm zeigen“ | Fordert Freigabe, dann ein Bildschirmfoto für das lokale Vision-Modell. |
| Mikrofon / Rechtsklick „Zuhören“ | Push-to-talk, lokale Deutsch-Erkennung (Vosk). Bricht laufende Sprachausgabe ab. |
| „KIKI“ sagen | Nur wenn das Weckwort eingeschaltet ist. Danach wird die nächste Äußerung zur Frage. |
| Rechtsklick „Sprechen beenden“ | Stoppt PipeWire, leert die Warteschlange und bricht laufende TTS-Synthese ab. |
| „Status anhängen“ | Uhrzeit, Akku, Netzwerk, Speicher **sichtbar** an die Nachricht hängen |
| Strg+. | Chat |
| Strg+, | Einstellungen |
| Strg+Q | Beenden |
| Rechtsklick „Coding-Session“ / Strg+Umschalt+C | Plan-First-Agent im registrierten Workspace |
| Rechtsklick „PC-Steuerung“ / Strg+Umschalt+P | Kontrollfenster für sieben feste, einzeln freizugebende PC-Aktionen |
| Rechtsklick „Workspaces“ | Git-Ordner zur Allowlist hinzufügen oder entfernen |
| Chat „Coding-Session“ | Entwurf/letzte Nutzerzeile in die Coding-Session übernehmen |
| Chat-Kopf „PC-Steuerung“ | Öffnet nur das Kontrollfenster; führt noch keine Aktion aus |
| Sprache „KIKI, PC-Steuerung öffnen“ | Öffnet nur das Kontrollfenster; jede Wirkung braucht weiterhin Klick und Freigabe |

## Konfiguration

Datei: `~/.config/kiki/config.toml` (wird beim ersten Speichern angelegt). Vorlagen: `src/kiki/config/defaults.toml`.

Wichtige Schlüssel:

- `ai.provider` = `ollama` \| `kiki_harness` \| `openai_compatible`
- `ai.ollama.base_url`, `ai.ollama.model` (Default `qwen3-vl:4b`)
- `ai.ollama.num_ctx` — Kontextfenster (Default 8192; Ollamas eigener Default von
- `ai.ollama.think` — Denkmodus des Modells (Default `false`)
- `ai.ollama.suppress_thinking` — füllt einen geschlossenen, leeren Denkblock vor,
  damit das Modell gar keinen öffnen kann. Default `true`; auf eine echte Frage
  gemessen **66,0 s → 3,7 s** bis zum ersten Wort. Modelle ohne Denkmodus
  antworten damit unverändert.
  4096 ist zu knapp, sobald Systemprompt, Gedächtnis und Verlauf zusammenkommen)
- `ai.kiki_harness.base_url`, `.model`, `.quantize` (`int4`/`int8`/`none`), `.slots`
- `persona.id` = `begleiterin` \| `assistenz` \| `knapp` \| `eigene`
- `persona.address` — Anrede, leer = keine
- `ai.openai_compatible.base_url`, `ai.openai_compatible.model`
- `ai.system_prompt` — nur die **Persönlichkeit**, wenn `persona.id = "eigene"`.
  KIKIs feste Regeln stehen nicht hier, sondern im Programm.
- `pet.scale`, `pet.click_through_idle`, `pet.always_on_top`
- `app.privacy_panic` — alle Integrationen aus
- `tools.model_tool_use` — KIKI darf Werkzeuge selbst aufrufen (Default `true`)
- `tools.autonomy` — `strict` (nur lesen) oder `balanced` (Default; lesen + deklarierte Steuerung)
- `tools.max_steps`, `tools.max_tool_calls` — Obergrenzen pro Zug
- `voice.wake.enabled` — Weckwort „KIKI" (Default `false`)
- `voice.wake.phrases`, `voice.wake.cooldown_ms`, `voice.wake.command_timeout_s`
- `watch.enabled`, `watch.speak` — von sich aus melden (beide Default `true`)
- `watch.quiet_start`, `watch.quiet_end` — Ruhezeit (Default 22:00–08:00)
- `watch.cooldown_s`, `watch.max_per_hour` — Wiederholungs- und Stundenlimit
- `watch.battery.percent`, `watch.disk.percent` — Warnschwellen
- `tts.enabled`, `tts.base_url` (`http://127.0.0.1:18765`), `tts.speaker` (`Serena`), `tts.language` (`German`)
- `workspaces.allowed_roots` — Coding-Workspaces nur unter diesen Roots (Default: `~/Projects`, `~/Code`, `~/Projekte`, `~/Dokumente/Projekte`). Nicht `$HOME` und nicht `/`.

Logs: `~/.local/state/kiki/kiki.log`  
Datenbank: `~/.local/share/kiki/kiki.sqlite3`

Umgebungsvariable `KIKI_DATA_DIR` überschreibt den Pfad zu Charakter und Icons.

## Wayland-Grenzen (GNOME)

- KIKI **kann nicht** selbst in die untere rechte Ecke springen. Nach dem ersten Start einfach hinziehen; GNOME merkt sich oft die Position der App.
- **Immer im Vordergrund** kann die App unter Wayland nicht setzen. Rechtsklick → „Fenstermenü“ oder Alt+Leertaste → *Immer im Vordergrund*.
- Transparente Pixel sind klickdurchlässig, die Figur selbst nicht.
- Optionaler Fallback: `GDK_BACKEND=x11 kiki` (XWayland) für klassisches Always-on-top.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Coding-Workspaces (Phase 1A)

Phase 1A bildet die Sicherheitsgrenze für die inzwischen darauf aufbauenden Coding-Funktionen: KIKI startet einen Coding-Agenten nur für ein **explizit registriertes Git-Repository**. Registry, Validierung, Git-Snapshot und SQLite-Persistenz bleiben dabei unabhängig von GTK und Agent-Prozessen.

Regeln:

- Nur Verzeichnisse **innerhalb** von `workspaces.allowed_roots` (nach Auflösen aller Symlinks).
- Das Home-Verzeichnis selbst und `/` sind als Root verboten.
- Der Pfad muss die **Git-Repository-Wurzel** sein (kein Unterordner).
- Symlinks, die aus einem erlaubten Root hinauszeigen, werden abgelehnt.
- Gespeicherte Pfade werden vor jeder vertrauenswürdigen Nutzung erneut gegen Root, Symlinks und Git-Toplevel geprüft. Nachträgliches Umleiten eines Pfads wird abgelehnt.
- KIKI-Funktionen akzeptieren keine nicht registrierten Pfade (`not_registered`). Die Registry ist jedoch keine Betriebssystem-Sandbox für beliebige externe Prozesse.
- „Workspace entfernen“ verlangt in der GTK-Oberfläche eine Bestätigung und deaktiviert nur den Allowlist-Eintrag. Repository und lokale Session-Historie bleiben erhalten.
- Git wird nur gelesen (`status`, `rev-parse`, `remote get-url`, `diff`) mit fester Argumentliste und reduzierter Umgebung. Repository-lokale Hooks, `fsmonitor`, externe Diffs und `textconv` sind deaktiviert; Git-Fehler gelten nicht als „sauber“.
- Remote-URLs werden nur als Metadaten gespeichert. Zugangsdaten, Query-Parameter und Fragmente werden vorher entfernt; eine Migration verwirft ältere unbereinigte Remote-Metadaten.

Registrieren geht über `WorkspaceRegistry` oder den Workspace-Manager. Die Bestätigung für das Entfernen liegt bewusst an der sichtbaren UI-/Tool-Grenze; die Persistenzschicht selbst löscht niemals Dateien.

## Coding-Agent (Phase 1B)

KIKI orchestriert **OpenCode** als getrennten Prozess, ersetzt ihn nicht.

KIKI benötigt dafür die **OpenCode Terminal CLI** mit `opencode run`. Das
gleichnamige Desktop-RPM allein stellt auf Linux nicht zwingend diesen Befehl
bereit. Der Komplettinstaller erkennt beide getrennt, lässt die Desktop-App
unangetastet und ergänzt die CLI nur nach sichtbarer Zustimmung.

- Standardmodus: `plan_first`. Eine Umsetzung braucht die `plan_session_id` einer erfolgreich abgeschlossenen Planung für denselben Workspace und unveränderten Aufgabentext.
- Profile: `observe` (lesen/planen), `develop` (Umsetzung und Tests nur nach Einzelfreigabe), `operator` (deaktiviert).
- OpenCode-Aufruf ist eine **feste** Argv-Vorlage (`opencode run [-m model] <prompt>`). Kein Shell-String aus dem LLM.
- Prozessgruppe + Timeout; Stop akzeptiert nur bekannte laufende Sessions und sendet SIGTERM an deren Gruppe.
- Environment ohne Secrets (`AWS_*`, `SSH_AUTH_SOCK`, API-Keys, …).
- Testprofile sind Namen (`python_pytest`, `node_npm_test`, …), kein freies Kommando. Stdout/stderr werden vollständig geleert; sichtbar und gespeichert bleiben höchstens 64 KiB mit Kürzungshinweis.
- Freigaben sind an Tool-ID **und** den Hash der vollständigen Parameter gebunden und nur einmal gültig.

Aktuelle Sicherheitsgrenze: `observe` wird durch Policy, Prompt und einen
Worktree-Fingerprint vor/nach dem Lauf kontrolliert, ist aber noch keine
OS-Sandbox. OpenCode-interne Tool-Anfragen werden angezeigt/protokolliert, aber
noch nicht einzeln blockierend freigegeben; die sichtbare Startfreigabe gilt für
die gesamte Umsetzungs-Session.

```toml
[agents]
opencode_binary = "opencode"
plan_first = true
```

OpenCode-Output wird als Text/JSON-Zeilen normalisiert (`message`, `plan`, `error`, …).

## Coding-Session (Phase 1C)

Rechtsklick auf KIKI → **Coding-Session** (Strg+Umschalt+C) oder **Workspaces**.

- Workspace-Allowlist verwalten (hinzufügen / entfernen, Repo bleibt auf der Platte).
- Aufgabe formulieren, Profil `observe` oder `develop`.
- **Plan erstellen** (lesen) und **Agent starten** (nur `develop`, unveränderter abgeschlossener Plan und Freigabekarte mit Tool + Parametern).
- Reiter: Plan, Agent-Output, Diff, Tests, Audit.
- Warnbanner bei uncommitted Änderungen.
- **Ordner öffnen** und **Tests starten** brauchen dieselbe gebundene Freigabe; ein zugeordneter Testlauf muss zum Workspace seiner Session gehören.
- TTS spricht nur Kurzstatus („Plan fertig.“), keine Logs oder Diffs.

Diff laden ist read-only (`git diff HEAD`). Es gibt keinen automatischen Commit oder Push.

## Kontrollierte PC-Steuerung (Phase 1D)

Rechtsklick auf KIKI → **PC-Steuerung**. Jede Wirkung zeigt Tool, Ziel,
Parameter, Risiko, Begründung und Effekt in einem eigenen Dialog. Die Freigabe
ist an exakt diese Parameter gebunden, nur einmal verwendbar und wird auditiert.

Verfügbar sind genau diese Aktionen:

- **Projektordner öffnen** — `xdg-open` auf den registrierten Workspace.
- **Terminal** — feste Launcher-Vorlage (`kgx` / `ptyxis` / `gnome-terminal` / `xdg-terminal-exec`), `cwd` = Workspace, kein `-c`.
- **Editor** — nur ein Allowlist-Editor und genau der Workspace-Pfad.
- **Datei öffnen** — `xdg-open` auf eine Datei, die nach Symlink-Auflösung im Workspace bleibt.
- **URL öffnen** — nur `http`/`https`, keine `file:`/`javascript:`/`data:`, keine Zugangsdaten in der URL.
- **Text kopieren** — höchstens 8.192 sichtbare Zeichen über GDK; keine simulierte Tastatureingabe.
- **Benachrichtigung** — genau eine lokale Gio-Benachrichtigung.

Zwischenablage- und Benachrichtigungsinhalte werden im Freigabespeicher
redigiert und im Audit nur durch Parameterhash und Längen-/Statusangabe
repräsentiert. Der Sprachbefehl kann ausschließlich das Kontrollfenster öffnen.
KIKI implementiert bewusst keine globale Maus-/Tastatursimulation, keine
unsichtbare GUI-Automation und keine freie Shell aus Chat- oder Modelltext.

## Phase 1E

- **Live-Diff** während einer laufenden Session (ca. alle 2 s, read-only `git diff HEAD`).
- **Editor** aus Allowlist (`code` / `codium` / `gnome-text-editor` / `gedit`), nur Workspace-Pfad, nach Freigabe.
- **Briefing** schreibt Teilaufgaben/Risiko/Akzeptanzkriterien ins Aufgabenfeld, startet keinen Agenten.
- **In den Chat** hängt eine lokale Session-Zusammenfassung in den Chat (klar gekennzeichnet).
- Podman/Distrobox/SSH-Runner existieren als Schnittstelle und sind **fail closed**.

## Weckwort „KIKI" (Phase 2B)

Statt zu klicken kannst du KIKI ansprechen. Sie hört auf **„KIKI"** — auch als
Teil von „Hey KIKI" oder „Hallo KIKI" — und nimmt dann die **nächste** Äußerung
als Frage entgegen:

```
„Hallo KIKI."          → KIKI wechselt in den Zuhör-Zustand
„Wie voll ist die Platte?“ → landet im Chat wie eine getippte Frage
```

Ein Befehl gilt als beendet, sobald du etwa eine Sekunde schweigst — es gibt
keine feste Aufnahmedauer und keinen zweiten Klick.

Einschalten: **Einstellungen → Privatsphäre → Sprache → „Auf ‚KIKI' hören"**,
oder `voice.wake.enabled = true`.

**Das Weckwort ist per Default aus, und zwar bewusst.** Ein dauerhaft offenes
Mikrofon ist deine Entscheidung, nicht etwas, das ein Update mitbringt. Wenn du
es einschaltest, gilt:

- Audio wird **nie** auf Platte geschrieben und verlässt den Rechner nicht.
- Im Wartezustand läuft lokale Spracherkennung durchgehend. Erkannter Text wird
  gegen das Weckwort geprüft und **sofort verworfen** — nichts davon wird
  gespeichert, geloggt oder an ein Modell geschickt. Nur die Äußerung *nach*
  dem Weckwort wird verwendet.
- Der Panic-Schalter und „Spracheingabe aus" schalten das Weckwort mit ab.
- Während KIKI spricht, hört sie nicht zu — sonst würde ihre eigene Stimme sie
  wecken.
- Die Figur zeigt sichtbar, wenn das Mikrofon offen ist.

Warum durchgehende Erkennung und nicht nur ein Schlüsselwort-Modell: Vosks
Grammatikmodus wurde gemessen und verworfen. Auf das Weckwort eingeschränkt
bildet das kleine deutsche Modell beliebige Sprache auf die Weckphrasen ab —
„der Schlüssel liegt auf dem Tisch" wurde zu `hey kiki`. Das ergab 6 Fehlalarme
bei 12 Negativsätzen; der offene Erkenner ergab 0. Details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Eigenes Weckwort: `voice.wake.phrases = ["computer"]`. Wörter, die das deutsche
Modell nicht kennt, werden beim Start abgelehnt statt still nie zu zünden.

## Selbst öffnen (Phase 2F)

Auf der Vertrauensstufe **trusted** öffnet KIKI Dinge selbst, ohne Freigabekarte:

```
„Mach mir das Projekt auf."      → Dateimanager im registrierten Workspace
„Öffne src/main.py im Editor."   → Editor aus der Allowlist
„Mach ein Terminal da auf."      → Terminal mit cwd = Workspace
„Öffne die GTK-Doku."            → Browser, nur http(s)
```

Einzuschalten unter **Einstellungen → Privatsphäre → Selbstständigkeit →
Vertrauensstufe**.

Die Lockerung ist eng gezogen, und das steckt im Code:

- **Nur registrierte Workspaces.** Jeder Pfad kommt aus der Registry, die bei
  **jedem** Aufruf Roots, Symlinks und Git-Wurzel neu prüft. Dein Downloads-
  Ordner ist kein Workspace — den kann sie nicht öffnen.
- **Dateien nur innerhalb.** `../` und absolute Pfade werden abgewiesen.
- **Terminal und Editor** nutzen feste Vorlagen aus einer Allowlist. Es gibt
  kein `-c` und keinen Shell-String — was das Modell schreibt, wird nie zum
  Kommando.
- **URLs** nur `http`/`https`, ohne Zugangsdaten. Kein `file:`, kein
  `javascript:`, kein `data:`.
- **Zwischenablage und Benachrichtigung bleiben draußen.** Die Zwischenablage zu
  ersetzen ändert deine Daten und behält deshalb die Freigabekarte.
- **Schreiben und externe Aktionen fragen auf jeder Stufe.** Das ändert `trusted`
  nicht.
- Panic schaltet alles ab.

Eine Einschränkung, die du kennen solltest: Der Agent-Loop begrenzt, wie viel ein
einzelner Zug öffnen kann (`max_tool_calls`, plus Wiederholungserkennung). Eine
zugübergreifende Bremse gibt es noch nicht — über mehrere Fragen hinweg könnte
KIKI theoretisch viele Fenster öffnen.

## Von sich aus melden (Phase 2E)

KIKI wartet nicht mehr nur darauf, angesprochen zu werden. Sie achtet auf Akku
und Speicherplatz und meldet sich, wenn etwas knapp wird — mit einer
Desktop-Benachrichtigung und, tagsüber, gesprochen:

> „Der Akku ist bei 9 Prozent. Zeit zum Anstecken."

**Sie meldet nur — handeln darf sie weiterhin nichts von allein.** Das steckt im
Code, nicht bloß in der Absicht: Eine Meldung ist ein Datentyp, der Text trägt
und sonst nichts. Kein Tool, kein Callback, keine Parameter. Jede Systemänderung
behält ihre Freigabekarte.

Wann sie den Mund hält:

| Situation | Benachrichtigung | Sprache |
|---|---|---|
| Warnung, tagsüber | ja | ja |
| Warnung, Ruhezeit (22:00–08:00) | nein | nein |
| Dringend, Ruhezeit | ja | nein |
| Du sprichst gerade mit ihr | ja | nein |
| Panic-Schalter | nein | nein |

Dazu meldet sie dieselbe Sache höchstens alle 30 Minuten und insgesamt nie öfter
als sechsmal pro Stunde — auch wenn etwas schiefläuft. Und sie meldet sich pro
Ereignis **einmal**: Ein Akku bei 19 % sagt einmal Bescheid, nicht jede Minute.
Erst wenn du lädst oder der Wert weiter fällt, meldet sie sich wieder.

Einstellbar unter **Einstellungen → Privatsphäre → Von sich aus melden**:
an/aus, laut oder still, und die Ruhezeit.

Aktuell beobachtet sie Akku (Warnung unter 20 %, dringend unter 10 %) und
Speicherplatz (Warnung ab 90 %, dringend ab 96 %). Kalender und Build-Läufe sind
noch nicht dabei.

## KIKIs eigener LLM-Harness (Phase 2H)

Statt Ollama kann KIKI ihre eigene Modell-Laufzeit fahren — weder Ollama noch
llama.cpp, sondern PyTorch/transformers unter KIKIs Kontrolle:

```bash
./scripts/setup-tts.sh   # legt die gemeinsame venv an (falls noch nicht da)
./scripts/setup-llm.sh   # Harness einrichten und starten
# danach in der Konfiguration: ai.provider = "kiki_harness"
```

Auf einer RTX 5060 Ti mit `Qwen/Qwen3-4B-Instruct-2507` in int4 (4,1 GB VRAM):

| | Ollama | KIKI-Harness |
|---|---|---|
| „Reverse Proxy einrichten?", 1. Token | 66,0 s | **0,18 s** |
| Werkzeugaufruf + Antwort | — | 2,2 s |

Der Unterschied ist nicht die Schnittstelle — die kostet 0,02 % — sondern die
Kontrolle: Der Harness sperrt die `<think>`-Token-IDs per `bad_words_ids`, das
Modell **kann** keinen Denkblock öffnen.

Was der eigene Harness sonst noch bringt:

- **Werkzeuge über das Chat-Template des Modells**, nicht in Prosa beschrieben.
  Aufrufe werden aus dem Token-Stream geparst, auch wenn `<tool_call>` über zwei
  Chunks verteilt ankommt.
- **Slots mit Priorität** — ein Hintergrundjob blockiert kein Gespräch.
- **VRAM-Budget**, geteilt mit dem TTS-Dienst statt darum zu kämpfen.

Die venv wird mit der Sprachausgabe geteilt: beide brauchen denselben
torch-CUDA-Stack, und transformers 4.57 kann Qwen3 bereits — es muss also nichts
unter dem laufenden TTS-Modell aktualisiert werden.

**Cloud bleibt offen.** Lokal ist das *Modell*, nicht die Werkzeuge: Ein
Tool-Handler darf einen Cloud-Dienst aufrufen (dann sinnvollerweise als
`EXTERNAL`, also mit Freigabekarte). Für Cloud-Modelle bleibt
`ai.provider = "openai_compatible"`.

**Echtes Batching** ist aktiv: Ein Forward-Pass bedient alle aktiven Sequenzen.
Gemessen auf einer RTX 5060 Ti mit Qwen3-4B int4 — gleiche Binary, gleiche Slots,
nur `--batch` unterscheidet sich:

| gleichzeitige Anfragen | seriell | gebatcht | Faktor |
|---|---|---|---|
| 1 | 2,81 s | 2,81 s | 1,00× |
| 2 | 5,65 s | 3,24 s | **1,74×** |
| 4 | 13,02 s | 3,97 s | **3,28×** |

Eine einzelne Anfrage zahlt nichts. `exclusive` (Code-Review) wartet weiterhin,
bis der Batch leer ist, statt einen Pass zu teilen.

## Persona (Phase 2D)

KIKIs Ton ist umschaltbar: **Einstellungen → Persönlichkeit → Persona**.

| Persona | Klingt wie |
|---|---|
| **Begleiterin** (Default) | Warm und ruhig, dezent verspielt. KIKIs bisheriger Ton. |
| **Assistenz** | Trocken, vorausschauend, sachlich. Ein Butler statt einer Freundin. |
| **Knapp** | Nur das Nötigste, oft ein Satz. |
| **Eigene** | Dein eigener Text. |

Dieselbe Frage („Welcher Prozess belegt Port 8080?"), drei Töne — 32, 70 und
7 Wörter. Unter **Anrede** kannst du eintragen, wie KIKI dich ansprechen soll;
leer lassen heißt: gar nicht.

**Der Ton ist von den Regeln getrennt.** Wahrheit, Werkzeugdisziplin,
Gedächtnisdisziplin und die Freigabepflicht stehen unter „Feste Regeln",
kommen aus dem Programm und hängen in **jedem** Ton am Prompt — auch an einer
eigenen Persona. Du kannst sie einsehen, aber nicht wegschreiben.

Das behebt einen Fehler, der stillschweigend wirkte: `ai.system_prompt` war ein
einzelner Wert, und eine gespeicherte Konfiguration überschrieb den Default
komplett. Wer KIKI vor Phase 2A eingerichtet hatte, bekam die später ergänzten
Werkzeug- und Gedächtnisregeln **nie** — sie waren in `config.toml` eingefroren.
Jetzt liefert das Programm die Regeln bei jedem Start neu. Ein vorhandener
eigener Prompt bleibt als Persona „Eigene" erhalten.

## Gedächtnis (Phase 2C)

KIKI kann sich Dinge über dich merken und nutzt sie in **späteren, völlig
getrennten** Unterhaltungen:

```
„Merk dir, dass ich Fedora 44 mit GNOME benutze.“
   → Freigabekarte zeigt den exakten Text → gespeichert

(neuer Chat, kein gemeinsamer Verlauf)
„Welche Distribution habe ich noch mal?“
   → „Du hast Fedora 44 mit GNOME.“
```

**Merken und Vergessen zeigen immer eine Freigabekarte** — in jeder
Vertrauensstufe, auch wenn du selbst darum gebeten hast. Was KIKI sich merkt,
prägt jede spätere Antwort, deshalb siehst du vorher den genauen Wortlaut. Der
kann von dem abweichen, was du gesagt hast. Nachschlagen ist lesend und läuft
ohne Dialog.

Alles Gemerkte steht in **Einstellungen → Gedächtnis**: einsehbar, einzeln
löschbar, und mit „Alles vergessen" komplett. Es liegt lokal in
`~/.local/share/kiki/kiki.sqlite3`.

Grenzen, damit das Gedächtnis nicht ausufert oder entgleist:

- Höchstens 400 Zeichen je Eintrag und 200 Einträge; Duplikate werden abgelehnt.
- In den Prompt gehen höchstens 40 Einträge und 2.000 Zeichen.
- Zeilenumbrüche und Steuerzeichen werden entfernt. Eine Erinnerung kann die
  Prompt-Struktur damit nicht nachbauen und sich nicht als Systemanweisung
  ausgeben — sie bleibt ein Listenpunkt in einem als Daten markierten Block.
- Der Panic-Schalter und „Integrationen aus" halten das Gedächtnis aus dem
  Prompt und nehmen dem Modell die Werkzeuge weg.

KIKI leitet nichts von allein ab: Es wird nur gemerkt, worum du ausdrücklich
bittest oder was du klar als dauerhaft über dich mitteilst — keine
Gesprächsinhalte.

## Werkzeuge im Chat (Phase 2A)

KIKI beantwortet Fragen wie „Wie voll ist die Platte und wie spät ist es?“
selbst, statt dich nach den Daten zu fragen: Sie ruft die passenden Werkzeuge
auf und formuliert aus dem Ergebnis eine Antwort. Im Chat zeigt eine Zeile über
der Antwort, welches Werkzeug lief (`⚙ Speicher …` → `✓ status_disk`), und der
gespeicherte Verlauf vermerkt es ebenfalls.

Die Grenzen bleiben dieselben wie bisher:

- Sichtbar für das Modell sind **nur** Tools, die im Code ausdrücklich
  `model_callable = true` tragen — aktuell die vier `status_*`-Tools. Alles
  andere ist auch dann Default Deny, wenn das Modell danach fragt.
- **Vertrauensstufe** (`tools.autonomy`, Einstellungen → Privatsphäre →
  Selbstständigkeit): `strict` lässt das Modell nur lesen, `balanced` (Default)
  zusätzlich die deklarierten Sicherheitssteuerungen. **Schreibende und externe
  Aktionen zeigen in jeder Stufe die Freigabekarte** — daran ändert die Stufe
  nichts.
- Ein Tool mit `auto_allow = false` (etwa das Bildschirmfoto) fragt immer, egal
  wer es anfordert.
- Der Panic-Schalter entfernt die Werkzeuge sofort aus der Sicht des Modells,
  auch mitten in einer laufenden Antwort.
- Pro Zug höchstens `max_steps` Modellrunden und `max_tool_calls` Aufrufe;
  identische Wiederholungen laufen nur einmal.
- Jeder Audit-Eintrag unterscheidet `[user]` von `[model]`.

Abschalten geht in den Einstellungen oder mit `tools.model_tool_use = false`;
dann verhält sich der Chat wieder wie vor Phase 2A.

## Sicherheit

- Keine Shell aus Modellantworten.
- Keine versteckten Systemdaten im Prompt. Die einzige automatische Ergänzung ist
  das Gedächtnis — und darin steht nur, was du selbst freigegeben hast und in den
  Einstellungen sehen kannst.
- Unbekannte Tools: Default Deny.
- Modellaufrufbare Tools: getrennte Allowlist (`model_callable`), Default Deny.
- Proaktive Meldungen tragen nur Text. KIKI meldet sich von selbst, handelt aber
  nie von selbst.
- Schreibende/externe Tools: Vorschau + Bestätigung + Audit-Log.
- Panic-Schalter in den Einstellungen.
- Coding-Workspaces: Default Deny außerhalb der Allowlist. Kein Zugriff auf ganz `$HOME`.
- PC-Steuerung: sieben deklarierte Aktionen, Observe-Profil, Vorschau, Einzelfreigabe und Audit; keine universelle Fernsteuerung.

## Charakter tauschen

Standard ist `kiki-adult-v3`: eine erwachsene, ruhiger gestaltete KIKI mit eigenen
Reaktionen für Zuhören, Denken, Sprechen, Freude, Überraschung, Schlaf, Fehler und
Benachrichtigungen. Das bisherige Paket `kiki` bleibt als Alternative enthalten.

Lege ein eigenes Paket unter `data/character/<id>/` mit `manifest.toml` und
PNG/WebP-Frames an und setze `character.id`. Der Renderer-Eintrag `frames` ist der
MVP; andere Renderer sind im Manifest vorbereitet, aber nicht implementiert.

## Lizenz

MIT. Die Figur KIKI ist eine eigens erzeugte Illustration und verwendet **kein** Fedora-Markenlogo.
