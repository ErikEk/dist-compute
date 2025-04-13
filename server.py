import socket
import pickle
import time

HOST = 'localhost'
PORT = 5555

# Training config (example for PyTorch model)
job_config = {
    "lr": [0.01, 0.001],
    "batch_size": [16, 32],
    "epochs": 5
}

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            conn.sendall(pickle.dumps(job_config))  # send config
            time.sleep(1)  # Let the client receive fully
            data = b""
            while True:
                packet = conn.recv(4096)
                print(packet)
                if not packet:
                    break
                data += packet

            result = pickle.loads(data)
            print("Received result from client:")
            print(result)

if __name__ == "__main__":
    run_server()