from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from .models import NetworkConfig, ZnibConfig

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "network": NetworkConfig,
    "znib": ZnibConfig,
}
