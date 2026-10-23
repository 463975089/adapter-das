import json

c = json.load(open("/models/dsv41f/config.json"))
for k in sorted(c):
    if any(s in k for s in ("index", "hc_", "hidden_size", "num_hidden", "sinkhorn", "head", "rope")):
        print(k, "=", c[k])
