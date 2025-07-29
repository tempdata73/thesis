from dataclasses import dataclass, field


@dataclass
class InstanceStats:
    obj: list[float] = field(default_factory=list)
    rhs: list[float] = field(default_factory=list)
    timed_out: list[bool] = field(default_factory=list)


@dataclass
class TimeStats:
    mu: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)


@dataclass
class StatsSchema:
    solver: str
    time: TimeStats = field(default_factory=TimeStats)
    instance: InstanceStats = field(default_factory=InstanceStats)

    def update(self, mu: float, sigma: float, obj: float, rhs: float, timed_out: bool):
        self.time.mu.append(mu)
        self.time.sigma.append(sigma)
        self.instance.obj.append(obj)
        self.instance.rhs.append(rhs)
        self.instance.timed_out.append(timed_out)
