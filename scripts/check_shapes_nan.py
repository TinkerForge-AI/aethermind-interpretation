import json, numpy as np
# path/to/session_events.embedded.json
ev = json.load(open("../aethermind-perception/chunks/session_20250805_162657/session_events.embedded.json")) 
t = np.array([e["embeddings"]["T"] for e in ev])
s = np.array([e["embeddings"]["S"] for e in ev])
f = np.array([e["embeddings"]["F"] for e in ev])
print(t.shape, s.shape, f.shape)
print(np.isnan(f).any(), "NaNs in F?")
