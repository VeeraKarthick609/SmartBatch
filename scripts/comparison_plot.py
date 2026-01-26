import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import signal
import sys

# --- Configuration ---
# To simulate "millions", increase DURATION and USERS.
# For demo/verification, keep short but high concurrency.
USERS = 500       
SPAWN_RATE = 50   
DURATION = "20s" 

SCENARIOS = {
    "Baseline (No Batching)": {"BS": 1, "WAIT": 0.0},
    "SmartBatch (Batch 32)":  {"BS": 32, "WAIT": 0.01}
}
LOCUST_FILE = "tests/locustfile.py"

def run_server(bs, wait):
    print(f"\n>>> Starting Server [BS={bs}, Wait={wait}]")
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(bs)
    env["MAX_WAIT_TIME"] = str(wait)
    
    # Run uvicorn
    proc = subprocess.Popen(
        ["venv/bin/uvicorn", "smartbatch.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.DEVNULL, # Mute stdout to clean logs
        stderr=subprocess.PIPE     # Keep stderr for errors
    )
    # Wait for startup
    time.sleep(5)
    if proc.poll() is not None:
        print("Error: Server failed to start.")
        print(proc.stderr.read().decode())
        sys.exit(1)
    return proc

def run_locust(name):
    clean_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    csv_base = f"results_{clean_name}"
    
    # Cleanup previous files
    for f in [f"{csv_base}_stats.csv", f"{csv_base}_stats_history.csv"]:
        if os.path.exists(f): 
            os.remove(f)

    print(f">>> Running Locust for {name} (Duration: {DURATION})")
    cmd = [
        "venv/bin/locust",
        "-f", LOCUST_FILE,
        "--headless",
        "-u", str(USERS),
        "-r", str(SPAWN_RATE),
        "--run-time", DURATION,
        "--host", "http://localhost:8000",
        "--csv", csv_base,
        "--csv-full-history"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Locust failed with code {e.returncode}")
    
    return f"{csv_base}_stats_history.csv"

def plot_comparison(results_map):
    print("\n>>> Generating Comparison Plots...")
    
    metrics = [
        ("Requests/s", "Throughput (RPS)", "throughput_comparison.png"),
        ("50%", "Median Latency (ms)", "latency_p50_comparison.png"),
        ("95%", "p95 Latency (ms)", "latency_p95_comparison.png")
    ]

    for col, ylabel, filename in metrics:
        plt.figure(figsize=(10, 6))
        has_data = False
        
        for name, csv_file in results_map.items():
            if not os.path.exists(csv_file):
                print(f"Warning: Missing data file {csv_file}")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                # Filter for the endpoint (ignoring Aggregated row if present mid-stream)
                # Locust history stores regular snippets.
                data = df[df["Name"] == "/predict"]
                if not data.empty:
                    plt.plot(data['Timestamp'], data[col], label=name, marker='.')
                    has_data = True
            except Exception as e:
                print(f"Error plotting {name}: {e}")

        if has_data:
            plt.xlabel('Time (s)')
            plt.ylabel(ylabel)
            plt.title(f'Comparison: {ylabel}\n(Users={USERS}, Duration={DURATION})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(filename)
            print(f"  -> Saved {filename}")
        else:
            print(f"  -> Skipping {filename} (No data)")
            
        plt.close()

def main():
    results = {}
    
    try:
        for name, config in SCENARIOS.items():
            server = run_server(config["BS"], config["WAIT"])
            try:
                csv_file = run_locust(name)
                results[name] = csv_file
            finally:
                print(">>> Stopping Server...")
                # Graceful shutdown dance
                os.kill(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("Force killing server...")
                    server.kill()
                    server.wait()
                time.sleep(2) # Cooldown betwen runs

        plot_comparison(results)
        print("\n>>> Done! Review the generated .png files.")
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
        # Ensure server is killed if loop broke
        if 'server' in locals() and server.poll() is None:
            server.kill()

if __name__ == "__main__":
    main()
