# Copyright © 2025 Christoph Schlager, TU Wien

from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from .models.input import GravityConfig, ImpactConfig, NetworkConfig, RoutingConfig

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "network": NetworkConfig,
    "gravity": GravityConfig,
    "impact": ImpactConfig,
    "routing": RoutingConfig,
}
