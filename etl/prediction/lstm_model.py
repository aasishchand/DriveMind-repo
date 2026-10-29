"""Small LSTM model used for multi-step vehicle trajectory prediction."""
from __future__ import annotations

import torch
from torch import nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 48,
                 future_steps: int = 30, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, future_steps * 2))
        self.future_steps = future_steps

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1]).reshape(-1, self.future_steps, 2)
