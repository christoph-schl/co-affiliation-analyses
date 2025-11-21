from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from .models import GravityConfig, NetworkConfig

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "network": NetworkConfig,
    "gravity": GravityConfig,
}
