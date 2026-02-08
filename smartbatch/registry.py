from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    handler: Callable
    version: str
    status: str = "active" # active, inactive

# Global registry: Model Name -> Checkpoint/Version -> ModelMetadata
# _registry[model_name][version] = ModelMetadata
_registry: Dict[str, Dict[str, ModelMetadata]] = {}

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
    If version is None, returns the Handler for the latest registered version (lexicographically max) 
    OR essentially just the first one marked active if we had status logic.
    For simplicity: max version string.
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
        # Default: Latest version (lexicographically)
        # In a real system, you'd track "active" or "default" tag.
        latest_version = max(versions.keys())
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
