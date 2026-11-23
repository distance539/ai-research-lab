import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;p=json.loads((R/'predictions.json').read_text())
f,ax=plt.subplots(figsize=(10,5.625),dpi=180)
for name,label in [('global_method','Global'),('group_method','Length groups')]:ax.plot(range(1,1001),p[name]['prefix'],label=label)
ax.axvline(800,color='gray',linestyle=':');ax.set(xlabel='Accepted reviews (exploration only)',ylabel='Error fraction',title='Frozen logits, same 1000 calibration reviews');ax.legend();f.tight_layout();f.savefig(R.parent/'images'/'risk_controls.png')
