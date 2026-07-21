from abc import ABC, abstractmethod
from typing import Iterable

from ..interpolate import InterpolationJob
from ..worker import InterpolationJobResult

class JobExecutor(ABC):

    @abstractmethod
    def map(self, func, jobs: Iterable[InterpolationJob]) -> Iterable[InterpolationJobResult]:
        ...