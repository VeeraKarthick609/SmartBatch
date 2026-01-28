from typing import Dict, Callable, Any, Optional

# Global registry: Model Name -> Async Handler Function
_registry: Dict[str, Callable] = {}

def register(name: str):
    """
    Decorator to register a batched function under a specific model name.
    
    Usage:
        @register(name="yolo")
        @batch(max_batch_size=8)
        async def yolo_inference(batch): ...
    """
    def decorator(func):
        if name in _registry:
            # We might want to allow overwriting, but warning is good
            pass 
        _registry[name] = func
        return func
    return decorator

def get_model(name: str) -> Optional[Callable]:
    return _registry.get(name)

def get_all_models() -> Dict[str, Callable]:
    return _registry

def reset_registry():
    """Clear the registry. Useful for testing."""
    _registry.clear()
