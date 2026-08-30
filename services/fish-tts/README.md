# kiki-fish-tts (Experiment, 2026-08-30)

Fish Audio **S2 Pro** als lokaler TTS-Dienst mit demselben HTTP-Vertrag wie
`kiki_tts_server` (`/health`, `/v1/synthesize` → WAV). Die Stimme ist ein
fester Referenz-Clip (Voice Cloning); mitgeliefert wurde ein Serena-Klon.

**Status: auf der 16-GB-Karte (RTX 5060 Ti) nicht neben LLM + Whisper
lauffähig — der Dienst ist gebaut und getestet, aber deaktiviert.**

## Messwerte (RTX 5060 Ti, 16 GB)

| Konfiguration | VRAM des Dienstes | Neben LLM (4 GB) + Whisper (0,8 GB) + Desktop (~2 GB) |
|---|---|---|
| fp16 (bf16-Load) | ~13 GB | OOM |
| int8 weight-only | ~9,1 GB | OOM (fehlten ~200 MB) |
| int4 weight-only | — | OOM, jeder Versuch ~3 min |

Grund: Das Komposit ist real ~6 Mrd. Parameter (4B Slow-AR + 400M Fast-AR +
Audio-Decoder + 262k-Vokabular-Einbettungen). „4B" bezieht sich nur auf den
Slow-AR-Kern.

## Erforderliche lokale Patches im fish-speech-Checkout

1. **CPU-first laden:** `init_model` baut das Modell unter
   `with torch.device("cpu")` und konvertiert *danach* mit
   `.to(device, dtype)` — die fp32-Ladepitze (17 GB) passt sonst auf keine
   Consumer-Karte.
2. **`max_length` begrenzen** (z. B. 3072): Der Default-KV-Cache von 32768
   Token kostet allein ~4,8 GB. Achtung: `generate_long` reserviert hart
   2048 Token Generierungsraum, der Prompt muss also in
   `max_seq_len − 2048` passen.
3. **`tokenizer_config.json` mitladen** — ohne sie scheitert
   `FishTokenizer` an der unbekannten Architektur `fish_qwen3_omni`.

## Aktivieren (nur mit ≥ 24 GB VRAM, wie von Fish empfohlen)

```bash
# Referenz vorbereitet unter ~/Modelle/test/s2-pro/serena-ref.{npy,txt}
systemctl --user disable --now kiki-tts.service   # Qwen freigeben
sed -i 's/--quant int4/--quant int8/' ~/.config/systemd/user/kiki-fish-tts.service
systemctl --user enable --now kiki-fish-tts.service
```

Der Dienst läuft dann auf Port 18765 (gleicher Vertrag wie Qwen), KIKI braucht
keine Änderung. Emotions-Tags (`[freundlich]`, `[kurze Pause]`) werden von S2
Pro direkt im Text unterstützt.

## Lizenz

Fish Audio Research License — **nicht-kommerziell**. Für den privaten
Desktop-Einsatz unproblematisch; Weiterverbreitung der Modelle/Outputs im
Produktkontext nicht.
