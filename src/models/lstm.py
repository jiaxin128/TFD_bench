# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class LSTM1D(nn.Module):
    def __init__(self, in_channels, num_classes, activation=F.relu,
                 hidden_size=32, num_layers=1, bidirectional=False, dropout_rate=0.0):
        super().__init__()
        self.activation = activation
        self.bidirectional = bidirectional
        lstm_out = hidden_size * (2 if bidirectional else 1)

        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, 16, 7, stride=4, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_out, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pre(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        out = torch.cat([h_n[-2], h_n[-1]], 1) if self.bidirectional else h_n[-1]
        out = self.dropout(out)
        out = self.activation(self.fc1(out))
        return self.fc2(out)