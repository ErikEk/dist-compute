import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define the same simple model for training
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Client to receive work from the server and train the model
def client():
    # Connect to the server
    host = 'localhost'
    port = 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        # Define the hyperparameters for the model
        config = {
            'lr': 0.01,
            'epochs': 5
        }

        # Send the configuration to the server
        data = pickle.dumps(config)
        s.sendall(data)
        time.sleep(10)
        # Receive the training results from the server (optional)
        data = s.recv(1024)
        print("Received results from server:", data)

if __name__ == "__main__":
    client()