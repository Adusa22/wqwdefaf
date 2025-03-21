import torch
import torch.nn as nn
import torch.optim as optim

class MechanicOptimizer:
    """
    Mechanic: A Learning Rate Tuner
    This optimizer wraps around a base optimizer (e.g., SGD, Adam) and dynamically tunes
    the learning rate using regret minimization techniques.
    """

    def __init__(self, base_optimizer, params, beta=(0.9, 0.99, 0.999), lambda_=0.01, s_init=1e-8, eps=1e-8):
        self.base_optimizer = base_optimizer(params, lr=1.0)  # Base optimizer (initial LR = 1)
        self.beta = beta  # Exponential moving average parameters
        self.lambda_ = lambda_  # Regularization factor
        self.s_init = s_init  # Initial learning rate scaling factor
        self.eps = eps  # Small value to prevent division by zero
        self.s = torch.tensor([s_init])  # Learning rate scaling factor
        self.v, self.r, self.m = 0, 0, 0  # Initialize tracking variables
        self.x_ref = [p.clone().detach() for p in params]  # Store initial parameters
        self.delta = [torch.zeros_like(p) for p in params]  # Accumulate updates

    def step(self, grads):
        """Perform a Mechanic optimizer step using gradient updates."""
        # Compute the tuning quantity
        h_t = sum((g * d).sum() for g, d in zip(grads, self.delta))

        # Update exponential moving averages
        self.m = max(self.beta[0] * self.m, h_t)
        self.v = self.beta[1] * self.v + h_t**2
        self.r = max(0, self.beta[2] * self.r - self.s * h_t)

        # Compute wealth factor and learning rate scale
        W_t = self.s_init * self.m + self.r
        self.s = W_t / (torch.sqrt(torch.tensor(self.v)) + self.eps)


        # Apply updates
        for p, g, d in zip(self.x_ref, grads, self.delta):
            d += g  # Accumulate gradient updates
            p.data = p + self.s * d  # Update model parameters

        # Perform optimizer step
        self.base_optimizer.step()
