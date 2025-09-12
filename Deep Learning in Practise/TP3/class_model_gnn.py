import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self, in_channels = 50, hidden_channels = 256, out_channels = 121):
        super(StudentModel, self).__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=4)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4)
        self.gat3 = GATConv(hidden_channels * 4, out_channels, heads=6, concat=False, dropout=0.1)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.elu(x)
        x = self.gat2(x, edge_index)
        x = self.elu(x)
        x = self.gat3(x, edge_index)
        return x


# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
