import subprocess

problem = "Maximum Coverage"
budget = 100
iterations = 10
datasets = ["Facebook", "Wiki", "Deezer", "Slashdot", "Twitter", "DBLP", "YouTube", "Skitter"]

processes = []

for dataset in datasets:
    cmd = [
        "python", "main.py",
        "--problem", problem,
        "--budget", str(budget),
        "--dataset", dataset,
        "--iterations", str(iterations)
    ]
    p = subprocess.Popen(cmd)
    processes.append((dataset, p))
    print(f"[{dataset}] Started")

# Wait for all to finish
for dataset, p in processes:
    p.wait()
    print(f"[{dataset}] Finished with code {p.returncode}")
