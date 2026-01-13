import torch

# ===== DEVICE =====
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU tidak tersedia!")
    print("Using GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda")

device = get_device()

# ===== MODEL DUMMY =====
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = MyModel().to(device)

# ===== DATA DUMMY =====
x = torch.randn(4, 10).to(device)
y = torch.randn(4, 1).to(device)

# ===== FORWARD =====
output = model(x)
print("Output:", output)
