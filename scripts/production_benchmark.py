import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import signal
import sys
import argparse
import threading
from datetime import datetime

def log(msg):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def heartbeat(stop_event):
    while not stop_event.is_set():
        time.sleep(30)
        if not stop_event.is_set():
            log("... Benchmark is still running ...")

# Configuration
DEFAULT_USERS = 500       
DEFAULT_RATE = 50         
DEFAULT_DURATION = "1h"   

# Scenarios to compare
SCENARIOS = {
    "Baseline (No Batching)": {"BS": 1, "WAIT": 0.0,  "PORT": 8001},
    "SmartBatch (Batch 32)":  {"BS": 32, "WAIT": 0.01, "PORT": 8002}
}

LOCUST_FILE = "tests/locustfile.py"

def cleanup_port(port):
    """Force kill anything on the port"""
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

def run_server(bs, wait, port):
    cleanup_port(port)
    log(f"Starting Server [BS={bs}, Wait={wait}, Port={port}]...")
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(bs)
    env["MAX_WAIT_TIME"] = str(wait)
    
    proc = subprocess.Popen(
        ["venv/bin/uvicorn", "smartbatch.main:app", "--host", "0.0.0.0", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    time.sleep(5)
    
    if proc.poll() is not None:
        log("!! Server crashed on startup !!")
        err = proc.stderr.read().decode()
        print(err if err else "No stderr output")
        raise RuntimeError("Server failed to start")
        
    return proc

def run_locust(name, users, rate, duration, port):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    csv_base = f"results_prod_{safe_name}"
    
    for f in [f"{csv_base}_stats.csv", f"{csv_base}_stats_history.csv"]:
        if os.path.exists(f): os.remove(f)

    log(f"Running Locust: {name} on Port {port}")
    log(f"    Users: {users}, Spawn: {rate}, Duration: {duration}")
    
    cmd = [
        "venv/bin/locust",
        "-f", LOCUST_FILE,
        "--headless",
        "-u", str(users),
        "-r", str(rate),
        "--run-time", duration,
        "--host", f"http://localhost:{port}",
        "--csv", csv_base,
        "--csv-full-history"
    ]
    
    try:
        # Start heartbeat
        stop_hb = threading.Event()
        hb = threading.Thread(target=heartbeat, args=(stop_hb,), daemon=True)
        hb.start()
        
        subprocess.run(cmd, check=True)
        
        stop_hb.set()
        
    except subprocess.CalledProcessError:
        stop_hb.set()
        log(f"Locust finished with error (or timeout).")
    except KeyboardInterrupt:
        stop_hb.set()
        log("Locust stopped by user.")
        
    return f"{csv_base}_stats_history.csv"

def plot_all(results_map):
    log("Generating Comparison Plots (X-axis: Cumulative Requests)...")
    
    # Metric Config: (ColumnName, Label, Filename, Y-Label)
    metrics = [
        ("Requests/s", "Throughput", "production_throughput.png", "RPS"),
        ("50%", "Median Latency", "production_latency_p50.png", "Latency (ms)"),
        ("95%", "p95 Latency", "production_latency_p95.png",  "Latency (ms)")
    ]

    for col, label, filename, ylabel in metrics:
        plt.figure(figsize=(10, 6))
        
        for name, csv_file in results_map.items():
            if not os.path.exists(csv_file):
                log(f"Warning: Missing data for {name}")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                data = df[df["Name"] == "/predict"].copy()
                if data.empty: continue
                
                # Create Cumulative Requests Column
                # Locust 'Total Request Count' is the cumulative count at that snapshot
                data['Cumulative Requests'] = data['Total Request Count']
                
                # Sorting ensures clean line, though time usually monotonic
                data = data.sort_values('Cumulative Requests')
                
                plt.plot(data['Cumulative Requests'], data[col], 
                         label=name, marker='.', markersize=2, alpha=0.8)
            except Exception as e:
                log(f"Error reading {csv_file}: {e}")

        plt.xlabel('Cumulative Requests Processed')
        plt.ylabel(ylabel)
        plt.title(f'Comparison: {label} vs Requests')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        log(f"saved {filename}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE)
    parser.add_argument("--duration", type=str, default=DEFAULT_DURATION)
    args = parser.parse_args()

    results = {}

    try:
        for name, config in SCENARIOS.items():
            server = None
            port = config["PORT"]
            try:
                server = run_server(config["BS"], config["WAIT"], port)
                csv = run_locust(name, args.users, args.rate, args.duration, port)
                results[name] = csv
            except RuntimeError as e:
                log(f"Skipping {name} due to server error: {e}")
            except Exception as e:
                log(f"Unexpected error in {name}: {e}")
            finally:
                if server:
                    log(f"Stopping Server on {port}...")
                    os.kill(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=5)
                    except:
                        server.kill()
                    
                cleanup_port(port)

        plot_all(results)
        
    except KeyboardInterrupt:
        log("Aborted.")

if __name__ == "__main__":
    main()
