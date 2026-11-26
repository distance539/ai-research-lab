import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;d=json.loads((r/'results.json').read_text())
fig,ax=plt.subplots(figsize=(10,5.625))
for name,color in [('global','#247caa'),('group','#dd9144')]:
 ax.plot([128,1000],[d['cells'][f'{n}_{name}']['nll'] for n in [128,1000]],marker='o',label=name,color=color)
ax.set(xlabel='Calibration rows (fixed nested prefix)',ylabel='Exploration NLL',title='Registered 2 x 2 ablation; same 1000 exploration reviews')
ax.legend();ax.grid(alpha=.2);fig.tight_layout();(r.parent/'images').mkdir(exist_ok=True);fig.savefig(r.parent/'images/ablation_results.png',dpi=180)
