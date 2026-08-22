from kiki.runners.unavailable import UnavailableRunner


class PodmanWorkspaceRunner(UnavailableRunner):
    def __init__(self) -> None:
        super().__init__("podman")
