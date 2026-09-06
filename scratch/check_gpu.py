import torch
p = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {p.total_memory / (1024**3):.2f} GB ({p.total_memory} bytes)")
print(f"CUDA Capable: {p.major}.{p.minor}")
