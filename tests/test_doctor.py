from __future__ import annotations

import asyncio
import json

from kiki.config.settings import Settings
from kiki.platform import doctor
from kiki.platform.doctor import (
    DoctorCheck,
    DoctorReport,
    DoctorStatus,
    _fedora_check,
    _is_loopback,
    _ram_check,
)


def test_fedora_44_is_ready_and_other_systems_are_not() -> None:
    ready = _fedora_check('ID=fedora\nVERSION_ID="44"\n')
    older = _fedora_check('ID=fedora\nVERSION_ID="43"\n')
    other = _fedora_check('ID=ubuntu\nVERSION_ID="26.04"\n')

    assert ready.status is DoctorStatus.READY
    assert older.status is DoctorStatus.LIMITED
    assert other.status is DoctorStatus.MISSING
    assert ready.required and older.required and other.required


def test_ram_check_reports_target_and_invalid_input() -> None:
    ready = _ram_check("MemTotal:       8388608 kB\n")
    limited = _ram_check("MemTotal:       4194304 kB\n")
    missing = _ram_check("")

    assert ready.status is DoctorStatus.READY
    assert "8.0 GiB" in ready.detail
    assert limited.status is DoctorStatus.LIMITED
    assert missing.status is DoctorStatus.MISSING


def test_loopback_detection_does_not_accept_external_or_lookalike_hosts() -> None:
    assert _is_loopback("http://127.0.0.1:11434")
    assert _is_loopback("http://localhost:18765")
    assert _is_loopback("http://[::1]:18770")
    assert not _is_loopback("https://api.openai.com/v1")
    assert not _is_loopback("https://localhost.example.com")
    assert not _is_loopback("not a url")


def test_external_llm_provider_is_reported_without_contact(
    monkeypatch,
) -> None:
    settings = Settings()
    settings.ai.provider = "openai_compatible"
    settings.ai.openai_compatible.base_url = "https://provider.example/v1"
    settings.tts.enabled = False

    def forbidden_provider(*_args, **_kwargs):
        raise AssertionError("external provider must not be constructed or contacted")

    monkeypatch.setattr(doctor, "create_provider", forbidden_provider)
    checks = asyncio.run(doctor._service_checks(settings))

    llm = next(check for check in checks if check.name == "llm")
    assert llm.status is DoctorStatus.READY
    assert "nicht kontaktiert" in llm.detail


def test_local_services_are_pinged_without_exposing_health_details(
    monkeypatch,
) -> None:
    from kiki.ai.provider import ProviderHealth
    from kiki.voice.stt_client import SttHealth
    from kiki.voice.tts_client import TtsHealth

    settings = Settings()

    class LocalProvider:
        async def ping(self, model: str) -> ProviderHealth:
            assert model == settings.ai.ollama.model
            return ProviderHealth(
                ok=True,
                detail="secret backend detail",
                models=(model,),
                selected_model_present=True,
            )

    async def healthy_tts(base_url: str, *, timeout: float) -> TtsHealth:
        assert base_url == settings.tts.base_url
        assert timeout == 1.5
        return TtsHealth(ok=True, ready=True, detail="internal detail")

    async def healthy_stt(base_url: str, *, timeout: float) -> SttHealth:
        assert base_url == settings.voice.stt_service
        assert timeout == 1.5
        return SttHealth(ok=True, ready=True, detail="internal detail")

    monkeypatch.setattr(doctor, "create_provider", lambda *_args: LocalProvider())
    monkeypatch.setattr(doctor, "tts_health", healthy_tts)
    monkeypatch.setattr(doctor, "stt_health", healthy_stt)

    checks = asyncio.run(doctor._service_checks(settings))

    assert [(check.name, check.status) for check in checks] == [
        ("llm", DoctorStatus.READY),
        ("tts_service", DoctorStatus.READY),
        ("stt_service", DoctorStatus.READY),
    ]
    assert all("detail" not in check.detail for check in checks)


def test_report_has_stable_machine_readable_shape() -> None:
    report = DoctorReport(
        status=DoctorStatus.LIMITED,
        checks=(
            DoctorCheck("fedora", DoctorStatus.READY, "Fedora 44", required=True),
            DoctorCheck("gpu", DoctorStatus.LIMITED, "VRAM unbekannt"),
        ),
    )

    payload = json.loads(report.to_json())

    assert payload == {
        "checks": [
            {
                "detail": "Fedora 44",
                "name": "fedora",
                "required": True,
                "status": "ready",
            },
            {
                "detail": "VRAM unbekannt",
                "name": "gpu",
                "required": False,
                "status": "limited",
            },
        ],
        "schema_version": 1,
        "status": "limited",
    }
    assert report.strict_ok


def test_strict_report_requires_all_required_checks_to_be_ready() -> None:
    report = DoctorReport(
        status=DoctorStatus.LIMITED,
        checks=(DoctorCheck("fedora", DoctorStatus.LIMITED, "Fedora 43", required=True),),
    )

    assert not report.strict_ok
