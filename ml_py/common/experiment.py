from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class Experiment(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def fancy_name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()


class ExperimentData:
    def __init__(self, experiment: Experiment):
        time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.base_dir = Path("./experiments").joinpath(experiment.name()).joinpath(time)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def subdir(self, name: str) -> Path:
        return self.base_dir.joinpath(name)

    def file(self, name: str) -> Path:
        return self.base_dir.joinpath(name)
