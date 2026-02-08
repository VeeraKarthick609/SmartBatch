import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import torch
import uvicorn
from pydantic import BaseModel, model_validator

try:
    from torchvision.models import resnet18
except ImportError as exc:
    raise RuntimeError(
        "torchvision is required for this example. "
        "Install it with: pip install -r .examples/requirements.txt"
    ) from exc

# Prefer local workspace package over any older installed smartbatch version.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbatch import batch, register
from smartbatch.main import app


WORKERS = int(os.getenv("SB_EXAMPLE_WORKERS", "2"))
if WORKERS < 1:
    raise ValueError("SB_EXAMPLE_WORKERS must be >= 1")


@dataclass
class WorkerBundle:
    model: torch.nn.Module
    device: torch.device
    lock: Lock


def _device_for_worker(worker_id: int) -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    return torch.device("cpu")


def _build_worker_bundle(worker_id: int) -> WorkerBundle:
    # weights=None avoids downloading checkpoints; useful for local testing.
    model = resnet18(weights=None)
    model.eval()
    device = _device_for_worker(worker_id)
    model.to(device)
    return WorkerBundle(model=model, device=device, lock=Lock())


WORKER_BUNDLES: Dict[int, WorkerBundle] = {
    worker_id: _build_worker_bundle(worker_id) for worker_id in range(WORKERS)
}


class ResNetInput(BaseModel):
    image: List[List[List[float]]]

    @model_validator(mode="after")
    def validate_image_shape(self) -> "ResNetInput":
        if len(self.image) != 3:
            raise ValueError("image must have 3 channels (CHW format)")
        if any(len(channel) != 224 for channel in self.image):
            raise ValueError("image height must be 224")
        if any(len(row) != 224 for channel in self.image for row in channel):
            raise ValueError("image width must be 224")
        return self


def _prepare_batch_tensor(batch: List[ResNetInput], device: torch.device) -> torch.Tensor:
    tensors = [torch.tensor(item.image, dtype=torch.float32) for item in batch]
    batch_tensor = torch.stack(tensors, dim=0)
    return batch_tensor.to(device)


def _run_resnet(batch: List[ResNetInput], worker_id: int) -> torch.Tensor:
    bundle = WORKER_BUNDLES[worker_id % WORKERS]
    batch_tensor = _prepare_batch_tensor(batch, bundle.device)
    with bundle.lock:
        with torch.inference_mode():
            logits = bundle.model(batch_tensor)
    return logits


@register(name="resnet18", version="v1")
@batch(
    max_batch_size=32,
    max_wait_time=0.02,
    input_schema=ResNetInput,
    max_queue_size=256,
    workers=WORKERS,
    target_latency=0.2,
)
def infer_resnet_v1(batch: List[ResNetInput], worker_id: int = 0) -> List[int]:
    """
    Returns top-1 class index for each item.
    """
    logits = _run_resnet(batch, worker_id)
    top1 = torch.argmax(logits, dim=1)
    return top1.cpu().tolist()


@register(name="resnet18", version="v2")
@batch(
    max_batch_size=32,
    max_wait_time=0.02,
    input_schema=ResNetInput,
    max_queue_size=256,
    workers=WORKERS,
    target_latency=0.2,
)
def infer_resnet_v2(batch: List[ResNetInput], worker_id: int = 0) -> List[Dict[str, Any]]:
    """
    Returns top-5 class indices and scores for each item.
    """
    logits = _run_resnet(batch, worker_id)
    probabilities = torch.softmax(logits, dim=1)
    scores, indices = torch.topk(probabilities, k=5, dim=1)

    outputs: List[Dict[str, Any]] = []
    for row_scores, row_indices in zip(scores, indices):
        outputs.append(
            {
                "top5_classes": row_indices.cpu().tolist(),
                "top5_scores": [float(f"{score:.6f}") for score in row_scores.cpu().tolist()],
            }
        )
    return outputs


if __name__ == "__main__":
    host = os.getenv("SB_EXAMPLE_HOST", "0.0.0.0")
    port = int(os.getenv("SB_EXAMPLE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
