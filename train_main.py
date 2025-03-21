import torch
import torch.nn as nn
import torch.optim as optim
from main import MechanicOptimizer  # Import Mechanic optimizer

# Define a simple feedforward neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Single-layer model: 10 inputs → 1 output

    def forward(self, x):
        return self.fc(x)

# Create model instance
model = SimpleModel()
criterion = nn.MSELoss()  # Loss function

# Initialize Mechanic with a base optimizer (e.g., Adam)
optimizer = MechanicOptimizer(optim.Adam, model.parameters())

# Simulated training loop
for epoch in range(10):  # Train for 10 epochs
    inputs = torch.randn(32, 10)  # Random batch of 32 samples
    targets = torch.randn(32, 1)

    optimizer.base_optimizer.zero_grad()  # Reset gradients
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, targets)  # Compute loss
    loss.backward()  # Backpropagate

    grads = [p.grad for p in model.parameters()]  # Extract gradients
    optimizer.step(grads)  # Mechanic step

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
