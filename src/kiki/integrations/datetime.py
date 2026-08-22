"""Local clock. No network."""

from __future__ import annotations

from datetime import datetime

from kiki.integrations.base import IntegrationSnapshot


class DateTimeIntegration:
    id = "datetime"
    title = "Uhrzeit"

    def snapshot(self) -> IntegrationSnapshot:
        now = datetime.now().astimezone()
        return IntegrationSnapshot(
            id=self.id,
            title=self.title,
            available=True,
            data={
                "iso": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "timezone": now.tzname() or str(now.tzinfo),
            },
        )
