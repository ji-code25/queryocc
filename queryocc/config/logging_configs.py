from dataclasses import dataclass


@dataclass
class LoggingConfig:
    bev_features: bool = True
