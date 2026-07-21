from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from ..interpolate import InterpolationJob
from ..worker import InterpolationJobResult

class JobExecutor(ABC):

    @abstractmethod
    def map(
        self,
        func: Callable[[InterpolationJob], InterpolationJobResult],
        jobs: Iterable[InterpolationJob],
    ) -> Iterable[InterpolationJobResult]:
        ...
