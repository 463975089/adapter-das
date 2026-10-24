import subprocess, sys, os
for i in range(8):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(i))
    r = subprocess.run([sys.executable, "-c", "import torch; f,t=torch.cuda.mem_get_info(0); print(round(f/2**30,1),\"G free\")"], capture_output=True, text=True, env=env)
    print(f"card{i}: {r.stdout.strip() or r.stderr.strip()[:80]}")
