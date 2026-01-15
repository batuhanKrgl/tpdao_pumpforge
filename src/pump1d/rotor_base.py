from dataclasses import dataclass


@dataclass
class RotorState:
    solved: bool = False


class RotorBase:
    def __init__(self) -> None:
        self.state = RotorState()

    def solve(self) -> None:
        raise NotImplementedError
