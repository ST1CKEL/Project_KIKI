"""Secret Service (libsecret / GNOME Keyring). Never write keys to disk."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

SCHEMA_NAME = "io.github.projectkiki.Kiki"
OPENAI_API_KEY = "openai_compatible/api_key"


class SecretStoreError(Exception):
    """Secret Service is missing or the operation failed."""


@runtime_checkable
class SecretStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def available(self) -> bool: ...


class MemorySecretStore:
    """In-memory store for tests. Not used in the GTK app."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._data = dict(initial or {})

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def available(self) -> bool:
        return True


class UnavailableSecretStore:
    def get(self, key: str) -> str | None:
        raise SecretStoreError("GNOME Secret Service ist nicht verfügbar.")

    def set(self, key: str, value: str) -> None:
        raise SecretStoreError("GNOME Secret Service ist nicht verfügbar. API-Keys werden nicht gespeichert.")

    def delete(self, key: str) -> None:
        raise SecretStoreError("GNOME Secret Service ist nicht verfügbar.")

    def available(self) -> bool:
        return False


class LibsecretStore:
    def __init__(self) -> None:
        import gi

        gi.require_version("Secret", "1")
        from gi.repository import Secret

        self._Secret = Secret
        self._schema = Secret.Schema.new(
            SCHEMA_NAME,
            Secret.SchemaFlags.NONE,
            {"key": Secret.SchemaAttributeType.STRING},
        )

    def available(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        try:
            value = self._Secret.password_lookup_sync(self._schema, {"key": key}, None)
        except Exception as exc:  # pragma: no cover - GI
            raise SecretStoreError(f"Keyring-Lesen fehlgeschlagen: {exc}") from exc
        return value

    def set(self, key: str, value: str) -> None:
        if not value:
            self.delete(key)
            return
        try:
            self._Secret.password_store_sync(
                self._schema,
                {"key": key},
                self._Secret.COLLECTION_DEFAULT,
                f"KIKI {key}",
                value,
                None,
            )
        except Exception as exc:  # pragma: no cover - GI
            raise SecretStoreError(f"Keyring-Schreiben fehlgeschlagen: {exc}") from exc

    def delete(self, key: str) -> None:
        try:
            self._Secret.password_clear_sync(self._schema, {"key": key}, None)
        except Exception as exc:  # pragma: no cover - GI
            raise SecretStoreError(f"Keyring-Löschen fehlgeschlagen: {exc}") from exc


def create_secret_store() -> SecretStore:
    try:
        return LibsecretStore()
    except Exception as exc:
        log.warning("Secret Service unavailable: %s", exc)
        return UnavailableSecretStore()
