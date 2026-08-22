# KIKI v2 – PET-Modell

Der kompakte Erstentwurf liegt als [KIKI-v2-canonical.png](design/KIKI-v2-canonical.png)
vor. Nach dem Nutzerfeedback ist der erwachsenere
[KIKI-v3-Entwurf](design/KIKI-v3-adult-concept.png) die bevorzugte Designrichtung:
reifere Gesichtszüge, ungefähr fünf Köpfe Körperhöhe und eine ruhige, souveräne
Ausstrahlung. Der daraus abgeleitete Pack `kiki-adult-v3` ist seit Version 0.3.0
der Standard; der bisherige Pack `kiki` bleibt als kompatible Alternative erhalten.

## Zielprofil

- quadratische, bei 200–260 px gut lesbare Silhouette;
- dunkelviolettes Haar, blau-weiße Jacke und cyanfarbener Hex-Core als feste Identität;
- echtes RGBA, einheitliches 512×512-Canvas und stabiler Anker bei `(256, 474)`;
- getrennte Basiszustände (`idle`, `listening`, `thinking`, `speaking`, `sleeping`,
  `paused`) und kurze Reaktionen (`blink`, `greet`, `happy`, `surprised`, `error`,
  `notification`);
- späterer Layer-Pack für Augen, Mund, Hände, Core-Glow und Schatten statt
  inkonsistenter vollständiger Ganzkörperbilder.

## Umsetzung

Der Legacy-Frame-Renderer nutzt die echte GTK-Frame-Zeit, verarbeitet große
Zeitdifferenzen effizient, stellt nach einer Reaktion den vorherigen Basiszustand
wieder her und cached Alpha-Klickregionen. Hover löst eine kurze Happy-Reaktion aus.

Der Erwachsenen-Pack enthält 13 transparente Produktionsframes für sämtliche zwölf
App-Zustände. Jeder Frame wird reproduzierbar freigestellt, auf ein 512×512-Canvas
normalisiert und auf 460 px sichtbare Maximalhöhe mit einheitlichem Fußanker gesetzt.
Das Manifest definiert eigenständige Clips für jeden Zustand; `thinking` und
`speaking` nutzen jeweils zwei ruhige Animationsframes.

Automatische Pakettests prüfen Manifest-Vollständigkeit, Pfade, 512×512-RGBA,
transparenten Rand und nichtleere Alpha-Masken. Der RPM-Smoke-Test lädt den Pack aus
dem tatsächlich installierten Paket und erwartet alle zwölf Clips.
