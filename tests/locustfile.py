import random
import os
import numpy as np
from locust import HttpUser, task, between, constant
from PIL import Image

# --- Image Loading Logic ---
IMAGE_DIR = "tests/data"

def load_and_process_images():
    images = []
    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: {IMAGE_DIR} not found.")
        # Fallback to random noise if no images
        return [np.random.rand(3, 224, 224).tolist()]

    print(f"Loading images from {IMAGE_DIR}...")
    for f in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, f)
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        try:
            with Image.open(path) as img:
                img = img.convert('RGB').resize((224, 224))
                # Normalize 0-1
                arr = np.array(img).astype(np.float32) / 255.0
                # HWC -> CHW (3, 224, 224)
                arr = np.transpose(arr, (2, 0, 1))
                images.append(arr.tolist())
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            
    if not images:
        print("No valid images found. Using random noise.")
        return [np.random.rand(3, 224, 224).tolist()]
        
    print(f"Successfully loaded {len(images)} images.")
    return images

# Load once globally
IMAGES = load_and_process_images()

class InferenceUser(HttpUser):
    wait_time = constant(0) # Hammer mode as per previous setting

    @task
    def predict(self):
        img_data = random.choice(IMAGES)
        self.client.post("/predict", json={"data": img_data})
