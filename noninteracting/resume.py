"""Resume highstat.py at one size: python resume.py L"""

import json, sys, time
from highstat import run_L, CONFIG
L = int(sys.argv[1]); n = dict(CONFIG)[L]
d = json.load(open("highstat.json"))
t = time.time(); d[str(L)] = run_L(L, n, seed=90210 + L)
json.dump(d, open("highstat.json", "w"), indent=1)
r = d[str(L)]
print(f"L={L} n={n}  eta={r['eta']['mean']:.4e} d={r['d']['mean']:.4e} "
      f"r2={r['r2']['mean']:.4e} rrand={r['rrand']['mean']:.4e}  [{time.time()-t:.0f}s]")
