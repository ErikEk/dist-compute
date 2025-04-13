import socket
import ray
import pickle
from ray import tune
import torch
import torch.nn as nn
import torch.optim as optim
import time
# Define a simple neural network for training
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        #time.sleep(1) # Simulate some training time
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Server to handle client connections and distribute tasks
def server():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define the server address and port
    host = 'localhost'
    port = 65432

    # Create a TCP/IP socket for communication
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()

        print("Server is listening for clients...")

        # Accept client connections and distribute work
        conn, addr = s.accept()
        with conn:
            print(f"Connected to {addr}")
            # Receive the training configuration (hyperparameters) from the client
            data = conn.recv(1024)
            config = pickle.loads(data)

            # Use Ray to parallelize the task across multiple clients
            result = ray.remote(train_model).remote(config)
            print("Waiting for clients to finish training...")
            results = ray.get(result)
            print("Results from clients:", results)

# Remote function to handle model training (distributed via Ray)
@ray.remote
def train_model(config):
    # Create a simple model and train it with the provided config
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    # Generate random data for training
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))

    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    accuracy = (output.argmax(dim=1) == y).float().mean().item()
    return accuracy, loss.item()

if __name__ == "__main__":
    server()