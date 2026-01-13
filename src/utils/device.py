import torch

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ GPU TIDAK TERDETEKSI!\n"
            "Pastikan:\n"
            "- Driver NVIDIA terinstall\n"
            "- PyTorch versi CUDA\n"
        )

    print("✅ Using GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda")
