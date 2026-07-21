from collections.abc import Callable, Iterable

from .base import JobExecutor
from ..interpolate import InterpolationJob
from ..worker import InterpolationJobResult

class SerialExecutor(JobExecutor):

    def map(
        self,
        func: Callable[[InterpolationJob], InterpolationJobResult],
        jobs: Iterable[InterpolationJob],
    ) -> Iterable[InterpolationJobResult]:
        for job in jobs:
            yield func(job)
