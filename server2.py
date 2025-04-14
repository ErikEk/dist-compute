# server.py
import socket
import pickle
import ray
import io
from aggregator import Aggregator
import threading
import time
import sys

ray.init()
aggregator = Aggregator.remote()

def print_while_waiting():
    while not data_received:
        print("Waiting for data...", flush=True)
        time.sleep(10)

data_received = False

def start_server(host='127.0.0.1', port=5001):
    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    print(f"[SERVER] Listening on {host}:{port}")

    # Start thread
    thread = threading.Thread(target=print_while_waiting)
    thread.start()
    data_received = False
    while True:
        conn, addr = s.accept()
        print(f"[SERVER] Connection from {addr}")
        data = b""
        while True:
            chunk = conn.recv(4096)
            print("Downloading data...")
            if not chunk:
                break
            data += chunk
        print("Download complete.")
        print("data size: " + str(sys.getsizeof(data)))
        data_received = True

        model_state = pickle.loads(data)
        buffer = io.BytesIO(model_state["state"])
        aggregator.receive_update.remote(buffer)

        conn.send(pickle.dumps({"status": "received"}))
        conn.close()
        
        # Wait for thread to finish
        thread.join()

if __name__ == "__main__":
    start_server()