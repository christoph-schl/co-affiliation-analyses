from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from .models.input import GravityConfig, NetworkConfig, PlotConfig

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "network": NetworkConfig,
    "gravity": GravityConfig,
    "plot": PlotConfig,
}
