import subprocess
import time
import pandas as pd
import os
import signal
import sys

# Configurations to test
# Format: (Batch Size, Wait Time)
CONFIGS = [
    (1, 0.0),    # Baseline: No batching
    (8, 0.01),   # Conservative
    (32, 0.01),  # Balanced
    (64, 0.05),  # High Throughput / High Latency
]

LOCUST_DURATION = "10s" # Short run for demo
LOCUST_USERS = 50
LOCUST_SPAWN = 10
RESULTS_FILE = "study_results.csv"

def run_server(batch_size, wait_time):
    print(f"--- Starting Server [BS={batch_size}, Wait={wait_time}] ---")
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(batch_size)
    env["MAX_WAIT_TIME"] = str(wait_time)
    
    # Start Uvicorn in background
    proc = subprocess.Popen(
        ["venv/bin/uvicorn", "smartbatch.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    # Wait for startup
    time.sleep(5) 
    return proc

def run_locust(batch_size, wait_time):
    csv_prefix = f"results_bs{batch_size}_w{wait_time}"
    cmd = [
        "venv/bin/locust",
        "-f", "tests/locustfile.py",
        "--headless",
        "-u", str(LOCUST_USERS),
        "-r", str(LOCUST_SPAWN),
        "--run-time", LOCUST_DURATION,
        "--host", "http://localhost:8000",
        "--csv", csv_prefix
    ]
    subprocess.run(cmd, check=True)
    return f"{csv_prefix}_stats.csv"

def parse_results(csv_file, batch_size, wait_time):
    df = pd.read_csv(csv_file)
    # Get the "Total" row or Aggregate
    # Usually the last row is Aggregated? No, locust saves final stats
    # We want the "Aggregated" Row or just taking the endpoint
    try:
        row = df[df["Name"] == "/predict"].iloc[0]
        return {
            "Batch Size": batch_size,
            "Max Wait (s)": wait_time,
            "RPS": row["Requests/s"],
            "p50 Latency (ms)": row["50%"],
            "p95 Latency (ms)": row["95%"],
            "Failures": row["Failure Count"]
        }
    except Exception as e:
        print(f"Error parsing {csv_file}: {e}")
        return None

def main():
    final_results = []
    
    for bs, wait in CONFIGS:
        # 1. Start Server
        server_proc = run_server(bs, wait)
        
        try:
            # 2. Run Locust
            csv_file = run_locust(bs, wait)
            
            # 3. Parse Data
            stats = parse_results(csv_file, bs, wait)
            if stats:
                final_results.append(stats)
                print(f"Result: {stats}")
                
        finally:
            # 4. Kill Server
            os.kill(server_proc.pid, signal.SIGTERM)
            server_proc.wait()
            time.sleep(2) # Cooldown

    # 5. Save & Print Summary
    df = pd.DataFrame(final_results)
    df.to_csv(RESULTS_FILE, index=False)
    print("\n=== FINAL RESULTS ===")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
