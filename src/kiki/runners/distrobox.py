from kiki.runners.unavailable import UnavailableRunner


class DistroboxWorkspaceRunner(UnavailableRunner):
    def __init__(self) -> None:
        super().__init__("distrobox")
