# 🐾 Project KIKI – Freundliches 2D-KI-Desktop-Pet

<div align="center">

<img src="docs/design/project-kiki-banner.png" alt="Project KIKI Banner" width="720" />

### **Dein intelligenter, lokaler KI-Begleiter für Fedora Linux (GNOME / Wayland)**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Fedora 44](https://img.shields.io/badge/Fedora-44-294172.svg?logo=fedora&logoColor=white)](https://fedoraproject.org)
[![GTK 4 / Libadwaita](https://img.shields.io/badge/GTK-4.22%20%7C%20Libadwaita-3584e4.svg?logo=gnome&logoColor=white)](https://gnome.org)
[![Ollama Supported](https://img.shields.io/badge/LLM-Ollama%20%7C%20Qwen3--VL-FF6B6B.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Inhaltsverzeichnis

- [Über Project KIKI](#-über-project-kiki)
- [Charakter-Design & Zustände](#-charakter-design--zustände)
- [Architektur](#-architektur)
- [Schnellstart & Installation](#-schnellstart--installation)
  - [1. Ein-Klick-Installer (Empfohlen)](#1-ein-klick-installer-empfohlen)
  - [2. Manuelle RPM-Installation](#2-manuelle-rpm-installation)
  - [3. Entwicklungsumgebung & Quellcode](#3-entwicklungsumgebung--quellcode)
- [Lokale KI & Sprach-Subsystem](#-lokale-ki--sprach-subsystem)
  - [Lokales Vision-LLM (Ollama)](#lokales-vision-llm-ollama)
  - [Spracherkennung (Vosk STT)](#spracherkennung-vosk-stt)
  - [GPU-Sprachsynthese (Qwen3-TTS)](#gpu-sprachsynthese-qwen3-tts)
- [Feature-Übersicht](#-feature-übersicht)
  - [Desktop-Pet & Interaktion](#desktop-pet--interaktion)
  - [Chat & Vision](#chat--vision)
  - [OpenCode Coding-Agent & Workspaces](#opencode-coding-agent--workspaces)
  - [Sicherheit, Tool-Policy & Panic-Button](#sicherheit-tool-policy--panic-button)
- [Dokumentation](#-weiterführende-dokumentation)
- [Entwicklung & Tests](#-entwicklung--tests)
- [Lizenz](#-lizenz)

---

## 🌟 Über Project KIKI

**KIKI** ist ein 2D-Desktop-Begleiter für Fedora Linux, der direkt auf dem Desktop lebt. Sie bietet:

- **Echte Desktop-Präsenz:** Ein transparentes, interaktiv verschiebbares Wayland-Fenster mit animierten Emotionen und Zuständen.
- **Lokale Privatsphäre:** Vollständig lokales Vision-LLM über Ollama (`qwen3-vl:4b` / `8b`), lokale Offline-Spracherkennung mit Vosk und lokales TTS. Keine Cloud-Pflicht!
- **Sichere Assistenz & Coding:** Integrierter OpenCode-Coding-Agent mit strikter Workspace-Isolation, Git-Root-Schutz und interaktiver Genehmigung.
- **Strikte Sicherheitsarchitektur:** Standardmäßig *Default Deny* für alle Systemaktionen. KIKI führt niemals ungefragt Befehle auf deinem Rechner aus.

---

## 🎨 Charakter-Design & Zustände

KIKI verfügt über ein detailreiches, handgezeichnetes Charakter-Design mit 12 normalisierten Zuständen auf einem 512×512 Canvas:

<div align="center">

<img src="docs/design/KIKI-v3-adult-concept.png" width="280" alt="KIKI Charakter-Design" />

</div>

### Emotionen & Animations-Zustände

KIKI reagiert visuell in Echtzeit auf das Gespräch, Systemzustände und Agent-Aktionen. Alle Sprites sind hochauflösend (512×512) auf transparentem Canvas normalisiert:

| **Idle** (Warten) | **Happy** (Freude) | **Thinking** (Nachdenken) | **Speaking** (Sprechen) |
|:---:|:---:|:---:|:---:|
| <img src="data/character/kiki-adult-v3/idle/00.png" width="130" alt="Idle" /> | <img src="data/character/kiki-adult-v3/happy/00.png" width="130" alt="Happy" /> | <img src="data/character/kiki-adult-v3/thinking/00.png" width="130" alt="Thinking" /> | <img src="data/character/kiki-adult-v3/speaking/00.png" width="130" alt="Speaking" /> |
| **Listening** (Zuhören) | **Sleeping** (Schlafmodus) | **Surprised** (Überrascht) | **Error** (Fehler / Not-Aus) |
| <img src="data/character/kiki-adult-v3/listening/00.png" width="130" alt="Listening" /> | <img src="data/character/kiki-adult-v3/sleeping/00.png" width="130" alt="Sleeping" /> | <img src="data/character/kiki-adult-v3/surprised/00.png" width="130" alt="Surprised" /> | <img src="data/character/kiki-adult-v3/error/00.png" width="130" alt="Error" /> |
| **Greeting** (Begrüßung) | **Notification** (Hinweis) | **Idle Blink** (Blinzeln) | **Status / Working** (Aktiv) |
| <img src="data/character/kiki-adult-v3/greet/00.png" width="130" alt="Greeting" /> | <img src="data/character/kiki-adult-v3/notification/00.png" width="130" alt="Notification" /> | <img src="data/character/kiki-adult-v3/idle_blink/00.png" width="130" alt="Idle Blink" /> | <img src="data/character/kiki-adult-v3/happy/00.png" width="130" alt="Working" /> |

---

## 🤖 Agentic Desktop-Assistenz

KIKI verwendet einen gemeinsamen **Assistant-Core** (`kiki.assistant`) für normalen Chat mit Modell-Werkzeugen und für den expliziten `/agent`-Entwicklerpfad. Dadurch laufen alle Agenten-Aufgaben kontrolliert, transparent und nachvollziehbar über denselben Sicherheitsweg:

```text
/agent erstelle eine Notiz mit "Projekt-Meilensteine prüfen"
```

<div align="center">

```
┌────────────────────────────────────────────────────────┐
│ ⏳ KIKI arbeitet …                         [Abbrechen] │
└────────────────────────────────────────────────────────┘
```

</div>

### Kernfunktionen des Assistant-Cores:

1. **Ein gemeinsamer Runner:** Chat und `/agent` verwenden denselben `AssistantRunner` für Streaming, Werkzeugschritte, Limits und Abbruch.
2. **Ein gemeinsamer Sicherheitsweg:** Jeder Werkzeugaufruf läuft über `ToolGateway`, `ToolPolicy`, Bestätigungsbroker und Audit. Panic- und Integrationsstatus werden unmittelbar vor einem Seiteneffekt erneut geprüft.
3. **Run-gebundener Statusbalken:** Während der Ausführung erscheint im Chatfenster ein kompakter Statusbalken mit Spinner und **`[Abbrechen]`-Button**.
4. **Sichere Bestätigung:** Schreibende und externe Aktionen verlangen außerhalb des expliziten Jarvis-Modus eine interaktive Genehmigung. Freigaben sind einmalig an Run, Tool-Aufruf, validierte Argumente und angezeigte Vorschau gebunden.
5. **Stabile Run-Identität (`run_id`):** Verspätete Callbacks, Freigaben oder Events eines alten Laufs können keinen aktuellen Lauf verändern.
6. **Harte Laufgrenzen:** Schrittzahl, Werkzeugaufrufe und Wiederholungen sind begrenzt; Protokollfehler enden sichtbar statt mit einer unvollständigen Antwort.

---

## 🏛 Architektur

Project KIKI trennt die Benutzeroberfläche strikt von der rechenintensiven KI-Logik, den Hintergrund-Threads und Subprozessen. Dadurch bleibt die GTK4-Oberfläche stets flüssig mit 60 FPS:

```mermaid
flowchart TD
    subgraph UI ["GTK4 / Libadwaita Interface (Main Thread)"]
        PetWindow["🐾 Pet Window (Transparent, Wayland)"]
        ChatWindow["💬 Chat, Statusbar & Cancel"]
        CodeWindow["💻 OpenCode Workspace Manager"]
        ConfirmDialog["🛡️ Confirmation Modal Dialog"]
        SettingsDialog["⚙️ Preferences Dialog"]
    end

    subgraph Bridge ["Async / UI Thread Bridge"]
        AsyncBridge["⚡ AsyncBridge (GLib.idle_add)"]
    end

    subgraph Core ["KIKI Core Engine (asyncio Thread)"]
        EventBus["📢 EventBus"]
        ChatService["🧠 ChatService (Gespräche & Streaming)"]
        RunService["🤖 RunService (Run-Lifecycle)"]
        AssistantRunner["⚙️ AssistantRunner (Chat + /agent)"]
        ToolGateway["🛡️ ToolGateway (Policy, Confirm, Audit)"]
        ToolRegistry["🔧 ToolRegistry (Allowlist)"]
        StateMachine["🎭 Character State Machine"]
        SpeechDirector["🎙️ Speech Director (Audio Controller)"]
    end

    subgraph Backends ["Lokale Services & Storage"]
        Ollama["🦙 Ollama (qwen3-vl:4b / 8b)"]
        VoskSTT["🎤 Vosk Offline STT (German)"]
        TTSMicroservice["🔊 Qwen3-TTS 0.6B (CUDA :18765)"]
        PipeWire["🎧 PipeWire / PulseAudio"]
        SQLiteDB["🗄️ SQLite WAL & Notes Workspace"]
    end

    ChatWindow -->|Chat oder /agent| AsyncBridge
    ConfirmDialog --> AsyncBridge
    PetWindow --> AsyncBridge
    CodeWindow --> AsyncBridge
    SettingsDialog --> AsyncBridge

    AsyncBridge <--> RunService
    AsyncBridge <--> ChatService
    AsyncBridge <--> StateMachine
    AsyncBridge <--> SpeechDirector

    RunService --> AssistantRunner
    ChatService --> AssistantRunner
    AssistantRunner --> ToolGateway
    ToolGateway --> ToolRegistry
    AssistantRunner --> Ollama

    ChatService --> Ollama
    ChatService --> SQLiteDB

    SpeechDirector --> VoskSTT
    SpeechDirector --> TTSMicroservice
    TTSMicroservice --> PipeWire
```

Weitere Details zur Architektur und Wayland-Besonderheiten findest du in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🚀 Schnellstart & Installation

### 1. Ein-Klick-Installer (Empfohlen)

Für Fedora Linux 44 steht ein vollständiger Ein-Datei-Installer (`.run`) bereit, der Paketintegrität, Abhängigkeiten und Setup automatisch ausführt:

```bash
# Installer ausführbar machen und starten
chmod +x KIKI-0.8.0-Fedora-44-x86_64.run
./KIKI-0.8.0-Fedora-44-x86_64.run
```

Der Installer führt interaktiv durch die optionale Einrichtung von **Ollama**, dem **Vosk-Sprachmodell** und der **OpenCode-CLI**.

---

### 2. Manuelle RPM-Installation

Das native RPM-Paket kann direkt über DNF installiert werden:

```bash
# RPM installieren
sudo dnf install ./dist/kiki-0.8.0-1.fc44.x86_64.rpm

# KIKI aus dem Terminal oder dem GNOME-App-Menü starten
kiki

# Fedora, Hardware und lokale Dienste ohne Seiteneffekte prüfen
kiki --doctor

# Für Setup-/CI-Skripte: Fehlercode bei fehlenden Kernvoraussetzungen
kiki --doctor --strict

# Maschinenlesbare Ausgabe
kiki --doctor --json
```

Der Doctor liest nur lokale Systeminformationen und prüft konfigurierte Dienste
ausschließlich auf Loopback-Adressen (`localhost`, `127.0.0.1`, `::1`). Externe
KI- oder TTS-Provider werden dabei weder kontaktiert noch authentifiziert.

---

### 3. Entwicklungsumgebung & Quellcode

Voraussetzungen auf Fedora 44:

```bash
# System-Abhängigkeiten installieren
sudo dnf install -y \
  rpm-build appstream desktop-file-utils systemd-rpm-macros \
  gdk-pixbuf2 git-core python3 python3-cairo python3-cffi \
  python3-gobject python3-httpx python3-pillow python3-pytest \
  espeak-ng

# Repository klonen
git clone https://github.com/ST1CKEL/Project_KIKI.git
cd Project_KIKI

# KIKI direkt aus dem Quellbaum starten
python3 -m kiki
```

---

## 🧠 Lokale KI & Sprach-Subsystem

### Lokales Vision-LLM (Ollama)

KIKI nutzt standardmäßig **Ollama** mit multimodalen Modellen (Text + Bildverständnis):

```bash
# 1. Ollama installieren & starten
sudo dnf install ollama
sudo systemctl enable --now ollama

# 2. Standard-Modell laden (qwen3-vl:4b)
kiki-setup-model
# Alternativ im Quellbaum: ./scripts/setup-local-model.sh
```

| Modell | Empfohlene Hardware | Einsatzzweck |
|---|---|---|
| **`qwen3-vl:4b`** *(Default)* | ≥ 8 GB RAM / 4 GB VRAM | Schneller Chat, Deutsch, Bildanalyse, OCR |
| **`qwen3-vl:8b`** *(Qualität)* | ≥ 16 GB RAM / 8 GB VRAM | Reifere Persona, tiefere Code- und Planungsfähigkeiten |
| **`gemma3:4b`** | ≥ 8 GB RAM | Exzellentes multilinguales Vision & Reasoning |

---

### Spracherkennung (Vosk STT)

- **Offline & Sicher:** Lokale Spracherkennung über Fedoras Vosk-0.3.50-Laufzeit.
- **Push-to-Talk:** Standardmäßig Leertaste/Button gedrückt halten.
- **Weckwort „KIKI“:** Optional in den Einstellungen aktivierbar.
- **Direktes Follow-up:** Nach einer per Weckwort gestarteten Antwort hört KIKI
  für genau eine weitere Äußerung zu. Bei Stille wartet sie wieder auf „KIKI“.
- **Automatisches Modell-Setup:**
  ```bash
  kiki --prepare-voice-model
  ```

---

### GPU-Sprachsynthese (Qwen3-TTS)

Für lebendige, natürlich klingende deutsche Sprachausgabe nutzt KIKI einen separaten **Qwen3-TTS 0.6B Microservice** (`CustomVoice Serena`):

Bei einer per Mikrofon gestellten Frage spricht KIKI standardmäßig höchstens
zwei Sätze beziehungsweise 300 Zeichen. Wurde die Antwort gekürzt oder wurden
sensible/technische Inhalte nicht vorgelesen, öffnet KIKI automatisch den Chat
mit dem unveränderten vollständigen Text. Beide Verhaltensweisen sind in den
Spracheinstellungen abschaltbar.

```bash
# TTS-Dienst einrichten (benötigt CUDA und Python 3.12)
kiki-setup-tts
# Alternativ im Quellbaum: ./scripts/setup-tts.sh

# Fallback: Sollte kein GPU-Dienst aktiv sein, nutzt KIKI automatisch espeak-ng!
```

Ausführliche Latenzanalysen und Architekturdetails: [docs/VOICE_SUBSYSTEM.md](docs/VOICE_SUBSYSTEM.md).

---

## ✨ Feature-Übersicht

### Desktop-Pet & Interaktion
- **Interaktives Bewegen:** Mit gedrückter linker Maustaste frei auf dem Bildschirm verschiebbar (`Gdk.Toplevel.begin_move`).
- **Klick-Durchlässigkeit:** Transparente Bereiche lassen Mausklicks durch, die Figur selbst bleibt interaktiv.
- **Kontextmenü (Rechtsklick):** Schneller Zugriff auf Chat, Einstellungen, Coding-Sessions, Ruhemodus und Beenden.

### Desktop-Steuerung & Jarvis-Modus
- **Medien & Audio:** MPRIS-Steuerung (Play/Pause/Next + Metadaten), Lautstärke und Stummschaltung über `pactl`.
- **Display & Sitzung:** Helligkeit über GNOME-/KDE-Sitzungsbus, Bildschirm sperren.
- **Anwendungen starten:** Index aller installierten `.desktop`-Einträge, Start über `gio launch` — ohne Shell, ohne Modelltext im Kommando.
- **Jarvis-Modus (experimentell):** Vertrauensstufe `jarvis` lässt KIKI auf allen Risikostufen ohne Rückfragen handeln; Hard-Deny-Liste, Panic-Schalter und Audit greifen weiterhin.
- **System & Netzwerk:** WLAN-Gerät schalten und Netzwerke anzeigen (NetworkManager), bestehende VPN-/WireGuard-Verbindungen per UUID verbinden und trennen, Ruhezustand/Neustart/Ausschalten über logind.
- **Freigegebene Routinen:** Wenn-Dann-Rezepte („Akku < 15 % → Aktion“), die einmal als komplette Karte bestätigt werden und dann ohne erneute Frage feuern — mit Cooldown, Panic-Stopp und Audit-Eintrag `routine`.

### Chat & Vision
- **Streaming Markdown:** Formatierter Text, Code-Blöcke mit Ein-Klick-Kopieren, LaTeX-Formeln und Tabellen.
- **Bildschirmfreigabe (Vision):** Screenshot über Wayland XDG Desktop Portal anfordern, um KIKI den Bildschirm oder ein bestimmtes Fenster zu zeigen.
- **Persistenter Verlauf:** SQLite WAL Datenspeicher unter `~/.local/share/kiki/` mit vollständiger Such- und Löschfunktion.

### OpenCode Coding-Agent & Workspaces
- **Workspace-Allowlist:** Exakte Begrenzung auf freigegebene Verzeichnisse mit Symlink-Auflösung und Schutz vor Pfadtraversierung.
- **Plan-First Ansatz:** Der Agent erstellt strukturierte Implementierungspläne vor jeder Code-Änderung.
- **Audit-Log:** Jeder Dateizugriff, Befehl und Subprozess wird revisionssicher protokolliert.

### Sicherheit, Tool-Policy & Panic-Button
- **Default Deny:** Gefährliche Systembefehle (`sudo`, freie Shells wie `sh -c "rm -rf ..."`) sind hart geblockt.
- **Interaktive Genehmigungskarten:** Bei Dateiänderungen oder externen Zugriffen fragt KIKI mit Diff-Vorschau um Erlaubnis.
- **Panic-Button:** Bricht sofort alle laufenden LLM-Streams, TTS-Ausgaben und Hintergrund-Subprozesse ab.

---

## 📚 Weiterführende Dokumentation

- [📘 Benutzerhandbuch (User Guide)](docs/GUIDE.md) – Detaillierte Anleitung für alle Funktionen und Tastaturkürzel.
- [🛠️ Entwicklerhandbuch (Developer Guide)](docs/DEVELOPER_GUIDE.md) – Bauen, Testen, RPM-Paketierung und Erstellung eigener Sprite-Packs.
- [🎙️ Voice Subsystem Architektur](docs/VOICE_SUBSYSTEM.md) – STT/TTS-Pipeline, Streaming-Pufferung und PipeWire-Integration.
- [🏛️ Systemarchitektur & Wayland-Design](docs/ARCHITECTURE.md) – Tiefgehende Architektur- und Sicherheitsdokumentation.
- [🎨 KIKI Charakter-Design](docs/CHARACTER_DESIGN.md) – Canvas-Spezifikationen und Animationsclips.

---

## 🧪 Entwicklung & Tests

Die Testsuite umfasst **mehr als 1.500 automatisierte Testfälle** für alle Schichten (Assistant-Core, Audio-Pipeline, UI-Event-Handling, Storage, Tool-Policy und Workspaces):

```bash
# Alle Tests ausführen
PYTHONPATH=src pytest

# Code-Linting mit Ruff
ruff check src tests services

# RPM-Paket und Installer im Projektverzeichnis bauen
./scripts/build-rpm.sh
./scripts/smoke-test-rpm.sh
./scripts/build-fedora-installer.sh
./scripts/smoke-test-installer.sh
```

---

## 📄 Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert – siehe die [LICENSE](LICENSE)-Datei für Details.
