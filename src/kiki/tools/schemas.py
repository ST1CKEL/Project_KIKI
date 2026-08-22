"""Tiny JSON-schema subset: type, required, properties, additionalProperties=false."""

from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Parameter validation failed."""


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_params(schema: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValidationError("parameters must be an object")
    if schema.get("type", "object") != "object":
        raise ValidationError("root schema must be an object")
    properties: dict[str, Any] = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    additional = schema.get("additionalProperties", False)
    extra = set(params) - set(properties)
    if extra and additional is False:
        raise ValidationError(f"unknown parameters: {', '.join(sorted(extra))}")
    for name in required:
        if name not in params:
            raise ValidationError(f"missing required parameter {name}")
    cleaned: dict[str, Any] = {}
    for name, value in params.items():
        spec = properties.get(name)
        if spec is None:
            if additional is False:
                raise ValidationError(f"unknown parameter {name}")
            cleaned[name] = value
            continue
        expected = spec.get("type")
        if expected:
            py = _TYPE_MAP.get(expected)
            if py is None:
                raise ValidationError(f"unsupported type {expected} for {name}")
            if expected == "integer" and isinstance(value, bool):
                raise ValidationError(f"{name} must be an integer")
            if not isinstance(value, py):
                raise ValidationError(f"{name} must be {expected}")
            if expected == "number" and isinstance(value, bool):
                raise ValidationError(f"{name} must be a number")
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            raise ValidationError(f"{name} must be one of {enum}")
        if isinstance(value, str):
            minimum = spec.get("minLength")
            maximum = spec.get("maxLength")
            if minimum is not None and len(value) < int(minimum):
                raise ValidationError(f"{name} must contain at least {minimum} characters")
            if maximum is not None and len(value) > int(maximum):
                raise ValidationError(f"{name} must contain at most {maximum} characters")
        cleaned[name] = value
    return cleaned
