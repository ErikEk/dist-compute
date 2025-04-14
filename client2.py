# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import socket
import pickle
import io
from model import SimpleNet

def train_local():
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.tensor(np.random.rand(16, 10), dtype=torch.float32)
    y = torch.tensor(np.random.randint(0, 2, size=(16,)), dtype=torch.long)
    print("Training local model...")
    for i in range(3*10000):  # local epochs
        if i % 10000 == 0:
            print(f"Local training step {i}")
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def send_update(host='127.0.0.1', port=5001):
    model_bytes = train_local()
    print("work done!")
    #print(model_bytes)
    payload = {"state": model_bytes}

    s = socket.socket()
    s.settimeout(3.0)
    s.connect((host, port))
    s.sendall(pickle.dumps(payload))
    response = b""
    try:
        print("[CLIENT] Model sent to server.")
        response = pickle.loads(s.recv(1024))
        #if not response:
        #    break

    except socket.timeout:
        print("Socket recv timed out and finished. FIX this in the server code.")
    if not response:
        print("[CLIENT] Received no response from server.")
    else:
        print("[CLIENT] Server response:", response)
    s.close()

if __name__ == "__main__":
    send_update()