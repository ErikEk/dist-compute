# aggregator.py
import ray
import torch
from model import SimpleNet

@ray.remote
class Aggregator:
    def __init__(self):
        self.global_model = SimpleNet()
        self.updates = []

    def receive_update(self, state_dict_bytes):
        state_dict = torch.load(state_dict_bytes)
        self.updates.append(state_dict)
        if len(self.updates) >= 2:  # Arbitrary aggregation trigger
            self.aggregate()

    def aggregate(self):
        avg_state = {}
        for k in self.updates[0]:
            avg_state[k] = sum(d[k] for d in self.updates) / len(self.updates)
        self.global_model.load_state_dict(avg_state)
        self.updates = []

    def get_global_model(self):
        return self.global_model.state_dict()