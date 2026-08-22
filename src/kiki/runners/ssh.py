from kiki.runners.unavailable import UnavailableRunner


class RemoteSSHRunner(UnavailableRunner):
    def __init__(self) -> None:
        super().__init__("ssh")
