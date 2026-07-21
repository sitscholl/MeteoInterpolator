from typing import Iterable

from .base import JobExecutor
from ..interpolate import InterpolationJob
from ..worker import InterpolationJobResult

class SerialExecutor(JobExecutor):

    def map(self, func, jobs: Iterable[InterpolationJob]) -> Iterable[InterpolationJobResult]:
        for job in jobs:
            yield func(job)