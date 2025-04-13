import socket
import pickle
import ray
from ray import tune
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import time

print(ray.__version__)
# Dummy model
class SimpleNN(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.fc(x)

# Ray training function
def train_model(config):
    # Fake dataset for demo
    X, y = make_classification(n_samples=500, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = SimpleNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        time.sleep(1)  # Simulate some training time
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        pred = torch.argmax(test_outputs, dim=1)
        acc = (pred == y_test).float().mean().item()
        print(f"Epoch {epoch+1}/{config['epochs']}, Accuracy: {acc:.4f}")

    return {"accuracy": acc, "loss": loss.item()}

# Receive job, do training, return result
def run_client():
    HOST = 'localhost'
    PORT = 5555

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.settimeout(2.0)
        data = b""
        print("Waiting to receive job config...")
        try:
            while True:
                print("Receiving data...")
                packet = s.recv(4096)
                print("Receiving data2...")
                if not packet:
                    break
                print(f"Got {len(packet)} bytes")
                data += packet
        except socket.timeout:
            print("Socket recv timed out and finished. FIX this in the server code.")

        config = pickle.loads(data)
        print("Received training job:", config)

        ray.init(ignore_reinit_error=True)
        print("Ray initialized.")
        print("Starting training...")
        result = tune.run(
            train_model,
            config={
                "lr": tune.choice(config["lr"]),
                "batch_size": tune.choice(config["batch_size"]),
                "epochs": config["epochs"]
            },
            num_samples=1,
            metric="accuracy",
            mode="max"
        )

        best = result.get_best_config(metric="accuracy", mode="max")
        print("Best config found:", best)

        s.sendall(pickle.dumps(best))

if __name__ == "__main__":
    run_client()