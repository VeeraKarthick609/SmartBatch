class SmartBatchError(Exception):
    """Base class for exceptions in SmartBatch."""
    pass

class OverloadedError(SmartBatchError):
    """Raised when the system is too overloaded to accept new requests."""
    pass
