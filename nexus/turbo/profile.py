from dataclasses import dataclass, field
import time
import psutil


@dataclass
class Profile:
    wall: float = field(default_factory=time.perf_counter)
    cpu: float = field(default_factory=time.process_time)

    def finish(self, tokens: int = 0) -> dict[str, float]:
        seconds = max(time.perf_counter() - self.wall, 1e-12)
        cpu_seconds = time.process_time() - self.cpu
        return {"wall_seconds": seconds, "cpu_seconds": cpu_seconds,
                "cpu_percent": 100 * cpu_seconds / seconds,
                "tokens_per_second": tokens / seconds,
                "rss_mb": psutil.Process().memory_info().rss / 2**20}
