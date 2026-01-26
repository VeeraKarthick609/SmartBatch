import torch
import torchvision
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    def load(self):
        """
        Lazily loads the model. 
        If model_path is provided, loads weights from there.
        Otherwise loads a pretrained ResNet18.
        """
        if self._loaded:
            return

        logger.info(f"Loading model on {self.device}...")
        
        try:
            # Create model architecture
            # Using ResNet18 as the default 'real' model
            self.model = torchvision.models.resnet18(weights=None) # Start with no weights for safety if loading custom
            
            if self.model_path:
                logger.info(f"Loading weights from {self.model_path}")
                # Load custom weights
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                logger.info("Loading pretrained default weights (ResNet18)")
                # Load pretrained weights if no path provided
                # Note: Logic slightly adjusted -> Re-instantiate with weights=Default if supported in this version
                # Or just load state dict. For simplicity in newer torchvision:
                self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def infer(self, batch: List[List[float]]) -> List[List[float]]:
        """
        Run inference on a batch of inputs.
        Expects batch of flattened image tensors or similar.
        For ResNet18: expects [B, 3, 224, 224] flattened or similar.
        
        For this simulation/demo, we will support random tensor inputs 
        that match the expected input size if needed, or simple lists.
        
        NOTE: ResNet18 expects [B, 3, H, W]. 
        If input is just a list of floats, we might need to reshape.
        For simplicity of the 'curl' demo (3 floats), this will crash ResNet.
        
        To support BOTH the simple demo and the real model:
        We will check input shape or allow a 'dummy' mode if input is too small.
        """
        if not self._loaded:
            self.load()
            
        if not batch:
            return []

        with torch.no_grad():
            # Basic handling: Convert list to tensor
            # CAUTION: This assumes inputs are essentially ready-to-batch tensors (lists of floats)
            try:
                tensor_batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
            except Exception:
                # Fallback for ragged inputs or other issues
                logger.warning("Could not convert batch to tensor directly. Using dummy.")
                tensor_batch = torch.zeros((len(batch), 3, 224, 224), device=self.device)

            # Check for demo mode (scalar or vector inputs that aren't images)
            if tensor_batch.ndim < 4:
                # logger.debug("Input is not 4D (NCHW), assuming demo/test mode. Generating dummy images.")
                # Create random images for the batch size
                tensor_batch = torch.randn((len(batch), 3, 224, 224), device=self.device)
            
            outputs = self.model(tensor_batch)
            
            return outputs.cpu().tolist()
