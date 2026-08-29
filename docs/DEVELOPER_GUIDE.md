# 🛠️ KIKI Entwicklerhandbuch (Developer Guide)

Dieses Dokument richtet sich an Entwickler, die an Project KIKI mitarbeiten, neue Funktionen hinzufügen, Charakter-Packs erstellen oder Pakete für Fedora Linux bauen möchten.

---

## 📑 Inhaltsübersicht

1. [Entwicklungsumgebung einrichten](#1-entwicklungsumgebung-einrichten)
2. [Code-Architektur & Designprinzipien](#2-code-architektur--designprinzipien)
3. [Testsuite & Qualitätssicherung](#3-testsuite--qualitätssicherung)
4. [Eigene Charakter-Packs erstellen](#4-eigene-charakter-packs-erstellen)
5. [Paketierung (RPM & Ein-Datei-Installer)](#5-paketierung-rpm--ein-datei-installer)
6. [Beitragskonventionen (Contributing)](#6-beitragskonventionen-contributing)

---

## 1. Entwicklungsumgebung einrichten

Project KIKI setzt auf **Fedora Linux 44** mit Python 3.13 oder 3.14.

### Systempakete installieren

```bash
sudo dnf install -y \
  rpm-build appstream desktop-file-utils systemd-rpm-macros \
  gdk-pixbuf2 git-core python3 python3-cairo python3-cffi \
  python3-gobject python3-httpx python3-pillow python3-pytest \
  espeak-ng ruff
```

### Projekt starten

```bash
# Im Projekt-Stammverzeichnis:
python3 -m kiki
```

---

## 2. Code-Architektur & Designprinzipien

### Grundregeln

1. **GI-freier Kern:** Die Kernlogik (`src/kiki/ai`, `src/kiki/agents`, `src/kiki/config`) darf keine direkten GTK/GObject-Abhängigkeiten haben. Dies ermöglicht isolierte, schnelle Unit-Tests.
2. **AsyncBridge:** Der GTK-Hauptthread darf niemals durch synchrone I/O blockiert werden. Asynchrone Tasks laufen im Hintergrund und senden Updates via `GLib.idle_add()` an die UI.
3. **Schichten-Trennung:**
   - `src/kiki/ui/`: GTK4 / Libadwaita Widgets und Fenster.
   - `src/kiki/application.py`: Anwendungs-Orchestrierung.
   - `src/kiki/ai/`: LLM-Provider, Chat-Service und Agenten-Loop.
   - `src/kiki/agents/`: OpenCode-Integration, Sandbox und Runner.
   - `src/kiki/character/`: Zustandsautomat, Renderer und Sprite-Lader.
   - `src/kiki/voice/`: Vosk STT, Qwen3-TTS Director & Playback Controller.

---

## 3. Testsuite & Qualitätssicherung

Die Testsuite deckt alle Komponenten mit mehr als 1.500 Unit- und Integrationstests ab.

```bash
# Alle Tests ausführen
pytest

# Spezifische Testmodule ausführen
pytest tests/test_chat_service.py
pytest tests/test_state_machine.py
pytest tests/test_local_runner.py
pytest tests/test_voice_route_flag.py

# Linting & Formatprüfung
ruff check src tests
```

---

## 4. Eigene Charakter-Packs erstellen

Jedes Charakter-Paket befindet sich unter `data/character/<pack-name>/` und wird über ein `manifest.toml` beschrieben.

### Canvas- und Frame-Spezifikationen

- **Dimensionen:** Exakt `512×512` Pixel im echten RGBA-Format (32-bit PNG mit Alpha).
- **Fußanker:** Die Figur muss vertikal an `(256, 474)` verankert sein.
- **Sichtbare Höhe:** Maximal `460` Pixel.
- **Transparenter Rand:** Mindestens 4 Pixel umlaufend transparent für sauberes Anti-Aliasing.

### Normalisierungs-Skript

Zur automatischen Freistellung und Zentrierung steht ein Skript bereit:

```bash
./scripts/normalize-character-frame.sh mein_rohes_bild.png data/character/mein-pack/idle/00.png
```

### Manifest-Aufbau (`manifest.toml`)

```toml
name = "mein-pack"
version = "1.0"
renderer = "frames"
default_state = "idle"
canvas_size = [512, 512]
anchor = [256, 474]

[clips.idle]
frames = ["idle/00.png", "idle/01.png"]
frame_duration_ms = 400
loop = true

[clips.speaking]
frames = ["speaking/00.png", "speaking/01.png"]
frame_duration_ms = 250
loop = true
```

---

## 5. Paketierung (RPM & Ein-Datei-Installer)

Project KIKI bietet automatisierte Skripte zur RPM-Erstellung und zum Bauen eines signaturfähigen Komplettinstallers.

```bash
# 1. Natives RPM-Paket bauen
./scripts/build-rpm.sh

# 2. RPM im temporären Verzeichnis auf Vollständigkeit testen
./scripts/smoke-test-rpm.sh

# 3. Den Ein-Datei-Installer (.run) generieren
./scripts/build-fedora-installer.sh

# 4. Den Installer auf Paketintegrität und SHA-256 prüfen
./scripts/smoke-test-installer.sh
```

Die fertigen Binärpakete landen im Verzeichnis `dist/`.

---

## 6. Beitragskonventionen (Contributing)

- **Git-Commits:** Bitte verwende [Conventional Commits](https://www.conventionalcommits.org/) (z. B. `feat:`, `fix:`, `docs:`, `chore:`).
- **Formatierung:** Vor jedem Commit sicherstellen, dass `ruff check src tests` und `pytest` fehlerfrei durchlaufen.
- **Sicherheit:** Niemals Passwörter, API-Schlüssel oder private Test-Zertifikate im Git einchecken (`.gitignore` beachten).
