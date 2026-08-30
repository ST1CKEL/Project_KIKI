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

## 2.2 Desktop-Steuerung & Jarvis-Modus

KIKI steuert den GNOME- oder KDE-Desktop direkt über den Chat oder per Sprache:

| Funktion | Werkzeuge | Beispiele |
|---|---|---|
| **Medien** (MPRIS) | `media.status`, `media.play_pause`, `media.next`, `media.previous`, `media.stop` | „Was läuft gerade für Musik?“ · „Nächster Titel“ |
| **Lautstärke** | `audio.volume_get`, `audio.volume_set`, `audio.mute` | „Stell die Lautstärke auf 30 Prozent“ · „Stumm“ |
| **Helligkeit** | `display.brightness_get`, `display.brightness_set` | „Dimm das Display auf 60 Prozent“ |
| **Anwendungen** | `app.list`, `app.open` | „Öffne Firefox“ · „Starte den Rechner (Calculator)“ |
| **Steam-Spiele** | `steam.list_installed`, `steam.launch` | „Starte Hades“ · „Öffne Portal 2 über Steam“ |
| **Sitzung** | `session.lock` | „Sperr den Bildschirm“ |
| **Netzwerk** | `network.wifi_list`, `network.wifi_set`, `network.vpn_list`, `network.vpn_connect`, `network.vpn_disconnect` | „Welche WLANs gibt es?“ · „Schalt das WLAN aus“ · „Verbinde das VPN“ |
| **Energie** | `power.suspend`, `power.reboot`, `power.poweroff` | „Schlafmodus“ · „Fahr den Rechner runter“ |

Lesen (READ) und Steuern (CONTROL) führt KIKI in der Standardstufe `balanced`
ohne Nachfrage aus — auch WLAN/VPN-Schaltung und Ruhezustand. Neustart und
Ausschalten (WRITE) fragen außerhalb des Jarvis-Modus nach. Modellseitige lokale
Starts (`LAUNCH`) klappt die Stufe `trusted` auf. Eine eindeutige Direktanweisung
wie „Starte Firefox“ oder „Öffne Hades über Steam“ ist bereits die Freigabe:
KIKI löst den Namen deterministisch gegen den lokalen App- beziehungsweise
Steam-Index auf, bevor irgendein Modell beteiligt wäre. Zusammengesetzte oder
unklare Sätze bleiben im normalen Chatpfad. Externe Aktionen zeigen in jeder
Vertrauensstufe eine neue Freigabekarte.

### Jarvis-Modus (experimentell)

In den Einstellungen unter **Selbstständigkeit → Vertrauensstufe** kann
**„… + ausgewählte Schreibaktionen (jarvis)“** gewählt werden. Dann darf KIKI
die einzeln geprüften Jarvis-Schreibwerkzeuge ohne Karte ausführen. Was auch
dann weiter gilt:

- Die Hard-Deny-Liste (`sudo`, freie Shell, `rm`, …) bleibt hart blockiert.
- Der Panic-Schalter stoppt sofort alles, auch Jarvis-Aktionen.
- Jede externe Aktion braucht weiterhin eine aktuelle Bestätigung; externe
  Routinen laufen deshalb nicht unbeaufsichtigt.
- Jede Ausführung landet weiterhin im Audit-Log.
- Werkzeuge mit bewusster Bestätigungspflicht (Routinen anlegen/löschen,
  Zwischenablage, Gedächtnis) zeigen auch im Jarvis-Modus ihre Karte.

## 2.3 Routinen („Wenn–Dann“)

Routinen sind die einzige Form, in der KIKI ohne Aufforderung handelt. Sag im
Chat:

> „Erstelle eine Routine: Wenn der Akku unter 15 Prozent fällt, spiel eine
> Benachrichtigung.“

KIKI zeigt daraufhin eine Freigabekarte mit dem **kompletten Rezept** —
Auslöser, Werkzeug, Argumente und Abklingzeit. Erst deine Freigabe speichert
die Routine. Sie feuert später genau so und ohne erneute Frage, mit einem
Cooldown (Standard 30 Minuten) gegen Dauerfeuer. Der Panic-Schalter hält alle
Routinen sofort an.

Verwaltung: Einstellungen → Seite **Routinen** (Liste, Ein-/Ausschalten,
Löschen) oder im Chat über `routines.list`, `routines.toggle`, `routines.delete`.

Mögliche Auslöser heute: Akkustand im Entladebetrieb und belegter
Heimat-Speicher (je Prozent, `lt`/`gt`/`eq`).

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
- **Direkte Rückfragen:** Nach KIKIs Antwort kannst du innerhalb des kurzen Hörfensters direkt weitersprechen. Ohne Spracheingabe endet das Fenster automatisch; deaktivierbar unter **Privatsphäre → Sprache**.
- **Sprachmodell wählen:** Über `[voice] stt_model` in der Konfiguration (`~/.config/kiki/config.toml`) steht zwischen zwei lokalen Modellen zur Wahl. Das kleine Standardmodell (`vosk-model-small-de-0.15`, ~45 MB) ist sparsam, verhört sich aber auf echten Stimmen häufig beim Namen „Kiki“ — das Weckwort reagiert dann nicht. Das große Modell (`vosk-model-de-0.21`) erkennt Sprache und Weckwort deutlich zuverlässiger, kostet aber einen ~1,9-GB-Download und beim Laden mehrere GB RAM.
- **Offline-Sicherheit:** Die Spracherkennung läuft zu 100 % lokal über das deutsche Vosk-Modell. Keine Sprachdaten verlassen deinen Rechner.

### Sprachausgabe (TTS)
- **Qwen3-TTS GPU-Dienst:** Erzeugt flüssige, warm klingende deutsche Sprache mit Satz-Streaming.
- **Kompakte Sprachantwort:** Auf Mikrofonfragen liest KIKI höchstens zwei Sätze bzw. 300 Zeichen vor. Gekürzte sowie aus Datenschutzgründen ausgelassene Details erscheinen vollständig im automatisch geöffneten Chat.
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
| **Routinen-Ursprung** | Jede Aktion trägt ihren Ursprung (`[user]`/`[model]`/`[routine]`) im Audit — nachvollziehbar, was ein Klick, was eine Modellentscheidung und was eine Routine war. |
| **Panic-Button (🚨)** | Ein Klick auf das Not-Aus-Symbol (oder Tastenkombination) bricht sofort: <br>• Alle laufenden KI-Generierungen ab <br>• Die Sprachausgabe ab <br>• Alle vom Agenten gestarteten Prozesse ab |

---

## 7. Einstellungen & Personas

Über das Einstellungsmenü (`Adw.PreferencesDialog`) kannst du KIKI anpassen:

- **Modell & Backend:**
  - Auswahl zwischen lokalem Ollama (`qwen3-vl:4b`, `qwen3-vl:8b`, `gemma3:4b`) oder OpenAI-kompatiblen Endpunkten.
  - Eingabe und sichere Speicherung von API-Keys im **GNOME Keyring** (über `libsecret`).
- **Erscheinungsbild:**
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
