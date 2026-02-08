from typing import Dict, Callable, Optional, List, Tuple
from dataclasses import dataclass
import re

@dataclass
class ModelMetadata:
    handler: Callable
    version: str
    status: str = "active" # active, inactive

# Global registry: Model Name -> Checkpoint/Version -> ModelMetadata
# _registry[model_name][version] = ModelMetadata
_registry: Dict[str, Dict[str, ModelMetadata]] = {}


def _version_sort_key(version: str) -> Tuple[Tuple[int, object], ...]:
    """
    Natural sort key for versions such as v1, v2, v10.
    Numeric fragments are compared numerically.
    """
    parts = re.findall(r"\d+|[^\d]+", version)
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)

def register(name: str, version: str = "v1"):
    """
    Decorator to register a batched function under a specific model name and version.
    
    Usage:
        @register(name="yolo", version="v2")
        @batch(max_batch_size=8)
        async def yolo_inference(batch): ...
    """
    def decorator(func):
        if name not in _registry:
            _registry[name] = {}
            
        _registry[name][version] = ModelMetadata(handler=func, version=version)
        return func
    return decorator

def get_model(name: str, version: str = None) -> Optional[Callable]:
    """
    Get the model handler.
    If version is None, returns the handler for the latest registered version using
    natural ordering (e.g., v10 > v9).
    """
    if name not in _registry:
        return None
        
    versions = _registry[name]
    if not versions:
        return None
        
    if version:
        meta = versions.get(version)
        return meta.handler if meta else None
    else:
        # Default: Latest version (natural ordering, e.g., v10 > v9)
        # In a real system, you'd track "active" or "default" tag.
        latest_version = max(versions.keys(), key=_version_sort_key)
        return versions[latest_version].handler

def get_model_versions(name: str) -> List[str]:
    if name not in _registry:
        return []
    return list(_registry[name].keys())

def get_all_models() -> Dict[str, List[str]]:
    """Returns Dict[ModelName, List[Versions]]"""
    return {name: list(versions.keys()) for name, versions in _registry.items()}

def reset_registry():
    """Clear the registry. Useful for testing."""
    _registry.clear()
