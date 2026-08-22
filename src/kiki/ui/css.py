from __future__ import annotations

APP_CSS = """
window.kiki-pet {
  background-color: transparent;
  background-image: none;
  box-shadow: none;
  border: none;
  outline: none;
}
window.kiki-pet > * {
  background-color: transparent;
}
picture.kiki-sprite {
  background-color: transparent;
}

.kiki-bubble {
  padding: 8px 12px;
  border-radius: 16px;
  margin: 4px 8px;
}
.kiki-bubble.user {
  background-color: alpha(@accent_bg_color, 0.18);
}
.kiki-bubble.assistant {
  background-color: alpha(@window_fg_color, 0.06);
}
.kiki-bubble.error {
  background-color: alpha(@error_color, 0.16);
}

.kiki-code-block {
  border-radius: 10px;
  padding: 0;
  margin: 6px 0;
  background-color: alpha(@window_fg_color, 0.06);
}
.kiki-code-header {
  padding: 4px 8px;
}
.kiki-code-body {
  font-family: monospace;
  font-size: 0.92em;
  padding: 8px;
}

.kiki-input {
  padding: 8px;
}

.kiki-mono {
  font-family: monospace;
  font-size: 0.9em;
}

.kiki-approval {
  padding: 10px;
  border-radius: 12px;
  background-color: alpha(@warning_color, 0.12);
}
"""
