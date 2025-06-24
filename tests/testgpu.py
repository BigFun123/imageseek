import torch
import time
from ultralytics import YOLO

print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name('cuda'))  # Your GPU name

print(torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


x = torch.randn(1000, 1000).cuda()  # Allocates GPU memory
y = torch.matmul(x, x)
for _ in range(10):
    y = torch.mm(x, x)
    time.sleep(1)  # Long enough for nvidia-smi to catch it


# Confirm GPU availability
if torch.cuda.is_available():
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available. Using CPU only.")

IMAGE = "images/test.jpg"  # Replace with your own image file

print("\n--- Running on CPU ---")
model_cpu = YOLO("models/yolov8n.pt")  # Tiny model for fast testing
start = time.time()
results_cpu = model_cpu(IMAGE, device='cpu')
print("CPU Inference Time:", round(time.time() - start, 3), "seconds")

print("\n--- Running on GPU ---")
model_gpu = YOLO("models/yolov8n.pt")
start = time.time()
results_gpu = model_gpu(IMAGE, device=0)
print("GPU Inference Time:", round(time.time() - start, 3), "seconds")

# Show results
results_gpu[0].show()  # or use .save() to save the image

# Optional: check if tensors are really on GPU
print("\nTensor device:", results_gpu[0].boxes.data.device)

# wait for key press to close the image window
input("Press Enter to exit...")
