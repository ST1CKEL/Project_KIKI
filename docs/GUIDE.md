# 📘 KIKI Benutzerhandbuch (User Guide)

Willkommen zum umfassenden Benutzerhandbuch für **Project KIKI**! Diese Anleitung erklärt dir alle Funktionen, Bedienkonzepte, Tastenkombinationen und Einstellungsmöglichkeiten im Detail.

---

## 📑 Inhaltsübersicht

1. [Grundbedienung des Desktop-Pets](#1-grundbedienung-des-desktop-pets)
2. [Chat & Konversationen](#2-chat--konversationen)
3. [Vision & Bildschirmfreigabe](#3-vision--bildschirmfreigabe)
4. [Sprachsteuerung & Sprachausgabe](#4-sprachsteuerung--sprachausgabe)
5. [OpenCode Coding-Sessions & Workspaces](#5-opencode-coding-sessions--workspaces)
6. [Sicherheit, Berechtigungen & Panic-Button](#6-sicherheit-berechtigungen--panic-button)
7. [Einstellungen & Personas](#7-einstellungen--personas)
8. [Tastaturkürzel & Schnellzugriffe](#8-tastaturkürzel--schnellzugriffe)

---

## 1. Grundbedienung des Desktop-Pets

KIKI erscheint nach dem Start als transparente Figur auf deiner Arbeitsfläche.

<div align="center">
  <img src="design/KIKI-v3-adult-concept.png" width="220" alt="KIKI Desktop Pet" />
</div>

### Interaktionen mit der Maus

- **Linksklick:** Öffnet das Chatfenster oder bringt ein minimiertes Chatfenster in den Vordergrund.
- **Verschieben (Drag & Drop):** Halte die linke Maustaste gedrückt und ziehe KIKI an die gewünschte Position auf deinem Bildschirm.
- **Mauszeiger darüber bewegen (Hover):** KIKI freut sich kurz über deine Aufmerksamkeit (`happy`-Reaktion).
- **Rechtsklick (Kontextmenü):**
  - 💬 **Chat öffnen**
  - 💻 **Coding-Session...**
  - ⚙️ **Einstellungen**
  - 🌙 **Schlafmodus** (versetzt KIKI in den Ruhezustand)
  - ❌ **KIKI beenden**

### Klick-Durchlässigkeit (Click-Through)
KIKIs Fenster berechnet seine Klickregion dynamisch anhand der Alpha-Transparenz der Charakter-Sprites. Transparente Pixel lassen Klicks direkt an darunterliegende Desktop-Fenster durch, während die Figur selbst zuverlässig klickbar bleibt.

---

## 2. Chat & Konversationen

Das Chatfenster bietet eine moderne Benutzeroberfläche auf Basis von GTK4 und Libadwaita.

```
┌────────────────────────────────────────────────────────┐
│  💬 KIKI Chat                             [—] [□] [✕] │
├────────────────────────────────────────────────────────┤
│                                                        │
│  [KIKI]: Hallo! Wie kann ich dir heute helfen?         │
│                                                        │
│  [Du]: Kannst du mir bei Python helfen?               │
│                                                        │
│  [KIKI]: Natürlich! Zeig mir gerne deinen Code oder    │
│          starte eine Workspace-Session.               │
│                                                        │
├────────────────────────────────────────────────────────┤
│  [ 📎 Datei ] [ 📸 Bildschirm ]                        │
│  ┌─────────────────────────────────────────┐  [ 🎙️ ]   │
│  │ Nachricht an KIKI eingeben...           │  [ ▶ ]    │
│  └─────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────┘
```

### Funktionen im Chat
- **Echtzeit-Streaming:** KIKI antwortet im Token-Stream, sodass Antworten sofort lesbar sind.
- **Formatierte Ausgabe:** Volle Unterstützung für Markdown, Code-Syntaxhervorhebung, mathematische Formeln und Tabellen.
- **Code kopieren:** Jeder Codeblock verfügt über einen Schnellkopier-Button.
- **Chat-Verlauf & Verwaltung:**
  - Konversationen werden automatisch in einer lokalen SQLite-Datenbank (`~/.local/share/kiki/kiki.sqlite`) gespeichert.
  - Über das Menü oben links kann der aktuelle Chat geleert oder ein neuer Verlauf gestartet werden.

---

## 2.1 Agentic Desktop-Assistenz mit `/agent`

Neben normalen Chatgesprächen kannst du KIKI mit dem Präfix `/agent` direkte, kontrollierte Aufgaben zur Ausführung übergeben:

```text
/agent Erstelle eine Notiz mit dem Text "Einkaufsliste: Milch, Brot, Äpfel".
/agent Wie ist dein aktueller Status?
```

<div align="center">

```
┌────────────────────────────────────────────────────────┐
│ 💬 KIKI Chat                             [—] [□] [✕] │
├────────────────────────────────────────────────────────┤
│                                                        │
│  [Du]: /agent Erstelle eine Notiz mit dem Text "..."   │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ⚙️ KIKI arbeitet …                   [Abbrechen] │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
├────────────────────────────────────────────────────────┤
```

</div>

### Lebenszyklus eines Agent-Runs:

1. **Start:** KIKI initialisiert einen neuen isolierten Lauf mit eindeutiger `run_id`.
2. **Statusanzeige:** Ein Spinner und ein sprechendes Label informieren dich live:
   - `KIKI arbeitet …`
   - `KIKI führt eine Aufgabe aus …`
   - `KIKI wartet auf deine Bestätigung.`
3. **Abbrechen:** Ein Klick auf **`[Abbrechen]`** stoppt den Lauf sofort; verwaiste Aktionen werden nicht mehr ausgeführt.
4. **Bestätigungsdialog bei schreibenden Aktionen:**
   - Möchte KIKI eine Notiz anlegen (`create_note`), öffnet sich ein nativer Bestätigungsdialog.
   - Der Dialog zeigt Titel und Inhalt der geplanten Notiz.
   - Erst nach Klick auf **„Bestätigen“** wird die Notiz unter `~/.local/share/kiki/notes/` gespeichert.
   - Klickst du auf **„Abbrechen“**, wird der Vorgang ohne Seiteneffekte verworfen.
5. **Ergebnis:** Nach Abschluss wird der Statusbalken sauber ausgeblendet und KIKI liefert die formatierte Antwort im Chat.

---

## 3. Vision & Bildschirmfreigabe

KIKI kann Bilder und Bildschirmfotos analysieren, um dir bei visuellen Aufgaben, Fehlermeldungen oder Code auf dem Bildschirm zu helfen.

### So teilst du Inhalte:
1. **Datei anhängen (📎):** Wähle ein lokales Bild (`.png`, `.jpg`, `.webp`) über den Dateidialog aus.
2. **Bildschirm zeigen (📸):**
   - Klicke auf den Screenshot-Button.
   - Der standardmäßige Wayland Desktop Portal-Dialog öffnet sich.
   - Wähle aus, ob du den gesamten Bildschirm, ein bestimmtes Anwendungsfenster oder einen Ausschnitt freigeben möchtest.
   - Erst nach deiner ausdrücklichen Freigabe wird das Bild an das lokale Vision-Modell (`qwen3-vl`) übermittelt.

> [!IMPORTANT]
> **Privatsphäre-Garantie:** KIKI erstellt **niemals** heimlich Screenshots im Hintergrund. Jeder Screenshot erfordert deine aktive Freigabe über das XDG Desktop Portal.

---

## 4. Sprachsteuerung & Sprachausgabe

<div align="center">
  <img src="../data/character/kiki-adult-v3/listening/00.png" width="100" alt="Listening" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="../data/character/kiki-adult-v3/speaking/00.png" width="100" alt="Speaking" />
</div>

### Spracheingabe (STT)
- **Push-to-Talk:** Halte das Mikrofon-Symbol im Chat oder die konfigurierte Taste gedrückt, sprich deine Frage und lasse die Taste los.
- **Weckwort „KIKI“:** In den Einstellungen kann die permanente Weckworterkennung aktiviert werden (standardmäßig deaktiviert zur Ressourcenschonung).
- **Offline-Sicherheit:** Die Spracherkennung läuft zu 100 % lokal über das deutsche Vosk-Modell. Keine Sprachdaten verlassen deinen Rechner.

### Sprachausgabe (TTS)
- **Qwen3-TTS GPU-Dienst:** Erzeugt flüssige, warm klingende deutsche Sprache mit Satz-Streaming.
- **Barge-in / Sofortiger Abbruch:** Klickst du während des Sprechens auf Stop oder sendest eine neue Nachricht, stoppt KIKI die Audioausgabe augenblicklich und verwirft noch in Berechnung befindliche Audioschnipsel.
- **Systemstimmen-Fallback:** Ist kein GPU-Server aktiv, liest KIKI Texte automatisch über Fedoras Systemstimme (`espeak-ng`) vor.

---

## 5. OpenCode Coding-Sessions & Workspaces

Über das Menü **Coding-Session...** kannst du KIKI in ein bestimmtes Projektverzeichnis einladen.

```
┌────────────────────────────────────────────────────────┐
│  💻 KIKI Coding Workspace: ~/Projekte/MeinProjekt      │
├────────────────────────────────────────────────────────┤
│  📁 Workspace-Status: Registriert (Git Root validiert) │
│  🛡️ Autonomie-Level:  Balanced (Genehmigung erforderlich)│
│                                                        │
│  [ Plan ansehen ]  [ Diff prüfen ]  [ Ausführen ]      │
└────────────────────────────────────────────────────────┘
```

### Sicherheitsregeln für Workspaces:
1. **Registrierungspflicht:** KIKI greift ausschließlich auf Verzeichnisse zu, die du im Workspace-Manager explizit freigegeben hast.
2. **Plan-First:** Bevor Dateien geändert werden, generiert der Agent einen transparenten Implementierungsplan.
3. **Diff-Prüfung:** Änderungen werden als visuelles Diff dargestellt. Du entscheidest per Knopfdruck, ob die Datei geschrieben wird.
4. **Isolierte Ausführung:** Befehle und Tests laufen in einer kontrollierten Prozessgruppe mit bereinigten Umgebungsvariablen (ohne API-Keys oder Passwörter).

---

## 6. Sicherheit, Berechtigungen & Panic-Button

KIKI folgt einem strikten Sicherheitsmodell (**Default Deny**):

| Schutzmechanismus | Funktionsweise |
|---|---|
| **Hard-Deny-Liste** | Befehle wie `sudo`, `su`, freie Shell-Strings (`sh -c`) werden ausnahmslos blockiert. |
| **Audit-Log** | Alle Werkzeugaufrufe, Dateilesungen und Prozessstarts werden mit Zeitstempel protokolliert. |
| **Panic-Button (🚨)** | Ein Klick auf das Not-Aus-Symbol (oder Tastenkombination) bricht sofort: <br>• Alle laufenden KI-Generierungen ab <br>• Die Sprachausgabe ab <br>• Alle vom Agenten gestarteten Prozesse ab |

---

## 7. Einstellungen & Personas

Über das Einstellungsmenü (`Adw.PreferencesDialog`) kannst du KIKI anpassen:

- **Modell & Backend:**
  - Auswahl zwischen lokalem Ollama (`qwen3-vl:4b`, `qwen3-vl:8b`, `gemma3:4b`) oder OpenAI-kompatiblen Endpunkten.
  - Eingabe und sichere Speicherung von API-Keys im **GNOME Keyring** (über `libsecret`).
- **Erscheinungsbild:**
  - Wechsel zwischen dem Standard-Pack `kiki-adult-v3` und dem Chibi-Pack `kiki`.
  - Anpassung der Ankergröße und Fenster-Deckkraft.
- **Persona & Tonfall:**
  - Wähle den Charakterton (z. B. Hilfsbereit, Professionell, Humorvoll).
  - Die festen Sicherheitsregeln bleiben unabhängig von der Persona stets aktiv.
- **Autostart:**
  - Option „Bei Anmeldung starten“ richtet den sauberen XDG-Autostart ein.

---

## 8. Tastaturkürzel & Schnellzugriffe

| Tastenkombination | Aktion |
|---|---|
| <kbd>Enter</kbd> | Nachricht senden |
| <kbd>Shift</kbd> + <kbd>Enter</kbd> | Neue Zeile im Eingabefeld |
| <kbd>Ctrl</kbd> + <kbd>N</kbd> | Neuen Chat beginnen |
| <kbd>Ctrl</kbd> + <kbd>K</kbd> | Chat-Verlauf leeren |
| <kbd>Ctrl</kbd> + <kbd>,</kbd> | Einstellungen öffnen |
| <kbd>Escape</kbd> | Not-Aus / Laufende Generierung abbrechen (Panic) |
| <kbd>Alt</kbd> + <kbd>Leertaste</kbd> | GNOME-Fenstermenü des Pets aufrufen |
