import numpy as np, json
from numpy.linalg import norm
# path/to/session_events.embedded.json
ev = json.load(open("../aethermind-perception/chunks/session_20250805_162657/session_events.embedded.json"))
F = np.array([e["embeddings"]["F"] for e in ev])
F /= (norm(F, axis=1, keepdims=True) + 1e-9)
q = F[0]
sims = F @ q
# after computing sims
q_idx = 0
idx = [i for i in sims.argsort()[::-1] if i != q_idx][:5]
print([ (i, float(sims[i]), ev[i].get("annotations", {}).get("scene_type")) for i in idx ])
