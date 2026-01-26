import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import signal
import sys

# Configuration
BATCH_SIZE = 32
WAIT_TIME = 0.01
LOCUST_FILE = "tests/locustfile.py"
OUTPUT_CSV = "stress_test"
PLOT_FILE = "performance_plot.png"

# Simulation params
# We will simulate a ramp up to "millions" style load by aggressive user spawning
USERS = 200     # High concurrency
SPAWN_RATE = 10 
DURATION = "30s" # Keep it short for the session, but representative

def run_server():
    print(f"--- Starting Server [BS={BATCH_SIZE}, Wait={WAIT_TIME}] ---")
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(BATCH_SIZE)
    env["MAX_WAIT_TIME"] = str(WAIT_TIME)
    
    proc = subprocess.Popen(
        ["venv/bin/uvicorn", "smartbatch.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    time.sleep(5)
    return proc

def run_locust():
    print("--- Starting Locust Stress Test ---")
    # Clean previous
    for f in [f"{OUTPUT_CSV}_stats.csv", f"{OUTPUT_CSV}_stats_history.csv"]:
        if os.path.exists(f):
            os.remove(f)

    cmd = [
        "venv/bin/locust",
        "-f", LOCUST_FILE,
        "--headless",
        "-u", str(USERS),
        "-r", str(SPAWN_RATE),
        "--run-time", DURATION,
        "--host", "http://localhost:8000",
        "--csv", OUTPUT_CSV,
        "--csv-full-history" # Important for plotting over time
    ]
    subprocess.run(cmd, check=True)

def plot_results():
    print("--- Generating Plot ---")
    try:
        df = pd.read_csv(f"{OUTPUT_CSV}_stats_history.csv")
        
        # Filter for the endpoint
        data = df[df["Name"] == "/predict"]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot RPS
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RPS (Requests/sec)', color=color)
        ax1.plot(data['Timestamp'], data['Requests/s'], color=color, label='RPS')
        ax1.tick_params(axis='y', labelcolor=color)

        # Plot Latency on same graph (secondary axis)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Latency (ms)', color=color)  
        ax2.plot(data['Timestamp'], data['50%'], color=color, linestyle='--', label='Median Latency')
        ax2.plot(data['Timestamp'], data['95%'], color='orange', linestyle=':', label='p95 Latency')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('SmartBatch Load Test: RPS vs Latency')
        
        # Add a single legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()  
        plt.savefig(PLOT_FILE)
        print(f"Plot saved to {PLOT_FILE}")
        
    except Exception as e:
        print(f"Failed to plot: {e}")

def main():
    server = run_server()
    try:
        run_locust()
        plot_results()
    finally:
        os.kill(server.pid, signal.SIGTERM)
        server.wait()

if __name__ == "__main__":
    main()
