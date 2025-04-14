# worker.py
import ray
import torch
from model import SimpleNet

@ray.remote
class Trainer:
    def __init__(self):
        self.model = SimpleNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, x_data, y_data):
        x = torch.tensor(x_data, dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.long)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()