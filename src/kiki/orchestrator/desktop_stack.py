"""GI-free tool stack for the voice orchestrator.

Same registry, same policy, same gateway as the GTK app. A second executor
would be a second security story — we do not grow one.
"""

from __future__ import annotations

from dataclasses import dataclass

from kiki.config.settings import Settings
from kiki.harness.notes import NotesWorkspace
from kiki.harness.notes import spec as harness_note_spec
from kiki.harness.system_status import spec as harness_status_spec
from kiki.integrations.datetime import DateTimeIntegration
from kiki.integrations.disk import DiskIntegration
from kiki.integrations.networkmanager import NetworkManagerIntegration
from kiki.integrations.upower import UPowerIntegration
from kiki.paths import database_path, user_data_dir
from kiki.skills.desktop import DesktopPerceptionSkill
from kiki.skills.registry import SkillRegistry
from kiki.skills.system_status import SystemStatusSkill
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.storage.database import Database
from kiki.storage.memory_repository import MemoryRepository
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.tools.app_launch_tools import AppLaunchSkill, DesktopIndex
from kiki.tools.audio_tools import AudioControlSkill
from kiki.tools.audit import AuditLog
from kiki.tools.container_tools import ContainerSkill
from kiki.tools.direct_actions import DirectActionService
from kiki.tools.display_tools import DisplayControlSkill
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.launch_tools import DesktopLaunchSkill
from kiki.tools.media_tools import MediaControlSkill
from kiki.tools.memory_tools import MemorySkill
from kiki.tools.network_tools import NetworkControlSkill
from kiki.tools.policy import ToolPolicy
from kiki.tools.power_tools import PowerControlSkill
from kiki.tools.registry import ToolRegistry
from kiki.tools.session_tools import SessionControlSkill
from kiki.tools.steam_launch_tools import SteamIndex, SteamLaunchSkill
from kiki.workspaces.registry import WorkspaceRegistry


@dataclass
class DesktopStack:
    settings: Settings
    db: Database
    gateway: ToolGateway
    executor: ToolExecutor
    direct: DirectActionService
    memories: MemoryRepository

    def panic(self) -> bool:
        return bool(self.settings.app.privacy_panic)

    def integrations_active(self) -> bool:
        return self.settings.integrations_active()


def build_desktop_stack(
    settings: Settings,
    *,
    db: Database | None = None,
    vision_handler=None,
) -> DesktopStack:
    database = db or Database(database_path())
    tools = ToolRegistry()
    skills = SkillRegistry()
    disk_path = (settings.integrations.disk.extra or {}).get("path") or None
    upower = UPowerIntegration()
    disk = DiskIntegration(disk_path)
    skills.register(
        SystemStatusSkill(
            [DateTimeIntegration(), upower, NetworkManagerIntegration(), disk]
        )
    )
    skills.register(DesktopPerceptionSkill())
    memories = MemoryRepository(database)
    skills.register(MemorySkill(memories))
    workspaces = WorkspaceRegistry(
        WorkspaceRepository(database),
        allowed_roots=settings.workspaces.allowed_roots,
    )
    skills.register(DesktopLaunchSkill(workspaces))
    skills.register(MediaControlSkill())
    skills.register(AudioControlSkill())
    skills.register(DisplayControlSkill())
    app_index = DesktopIndex()
    steam_index = SteamIndex()
    skills.register(AppLaunchSkill(app_index))
    skills.register(SteamLaunchSkill(steam_index))
    skills.register(SessionControlSkill())
    skills.register(NetworkControlSkill())
    skills.register(PowerControlSkill())
    skills.register(ContainerSkill())
    skills.install_into(tools)
    tools.register(harness_status_spec())
    tools.register(harness_note_spec(NotesWorkspace(user_data_dir() / "notes")))
    if vision_handler is not None:
        from kiki.orchestrator.vision import vision_tool_spec

        tools.register(vision_tool_spec(vision_handler))
    executor = ToolExecutor(tools, ToolPolicy(settings.tools.autonomy), AuditLog(database))
    # Audit repository exists so agent sessions stay inspectable even without GTK.
    AgentAuditRepository(database)
    gateway = ToolGateway(
        executor,
        panic_check=lambda: settings.app.privacy_panic,
        integrations_check=settings.integrations_active,
    )
    direct = DirectActionService(gateway, app_index, steam_index)
    return DesktopStack(
        settings=settings,
        db=database,
        gateway=gateway,
        executor=executor,
        direct=direct,
        memories=memories,
    )
