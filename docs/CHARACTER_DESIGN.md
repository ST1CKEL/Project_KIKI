# 🎨 KIKI Charakter-Design & Canvas-Spezifikation

Das visuelle Erscheinungsbild von **KIKI** basiert auf einem reifen, detailreichen 2D-Design mit ruhiger, souveräner Ausstrahlung und lebendigen Animationen.

<div align="center">
  <img src="design/KIKI-v3-adult-concept.png" width="300" alt="KIKI Charakter" />
</div>

---

## 📐 Zielprofil & Spezifikationen

- **Silhouette:** Quadratisches Canvas, bei einer Skalierung von 200–260 px gestochen scharf und optimal lesbar.
- **Identitätsmerkmale:** Dunkelviolettes Haar, blau-weiße Hightech-Jacke und ein leuchtend cyanfarbener Hex-Core.
- **Canvas-Format:** Echtes RGBA (32-Bit PNG mit Alpha), einheitliches **512×512 Pixel** Canvas.
- **Fußanker:** Fester Ankerpunkt bei `(256, 474)` für konsistente Platzierung bei allen Animationen.
- **Sichtbare Maximalhöhe:** 460 px mit mindestens 4 px transparentem Rand für sauberes Anti-Aliasing.

---

## 🎭 Animations-Zustände & Manifest

KIKIs Zustandsautomat unterscheidet zwischen dauerhaften Basiszuständen und kurzzeitigen Reaktions-Clips:

| Typ | Zustände | Verhalten |
|---|---|---|
| **Basiszustände** | `idle`, `listening`, `thinking`, `speaking`, `sleeping`, `paused` | Laufen kontinuierlich bzw. bis zum nächsten Event. |
| **Reaktionen** | `blink`, `greet`, `happy`, `surprised`, `error`, `notification` | Werden einmalig abgespielt; danach kehrt KIKI in den vorherigen Basiszustand zurück. |

### Frame-Renderer & Normalisierung

- Der Frame-Renderer nutzt die native GTK-Frame-Clock für flüssige 60-FPS-Wiedergabe.
- Klickregionen werden anhand der Alpha-Maske gecacht, sodass transparente Bildbereiche klickdurchlässig bleiben.
- Hover über der Figur löst eine kurze, freundliche `happy`-Reaktion aus.
- Pakettests validieren automatisch die 512×512 RGBA-Integrität, Pfade und Alpha-Masken für alle 12 Zustände.
