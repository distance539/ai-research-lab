import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path(__file__).with_name('results.json').read_text());v=r['budgets'];fig,ax=plt.subplots(figsize=(10,5))
ax.plot([x['n'] for x in v],[x['exploration']['nll'] for x in v],'o-',label='Target calibration')
for k,col in [('source','orange'),('raw','gray')]:ax.axhline(r['baselines'][k]['nll'],label=k,color=col,linestyle='--')
ax.set(xscale='log',xlabel='Nested calibration sample count',ylabel='Exploration NLL',title='One frozen trajectory; no uncertainty interval');ax.legend();fig.tight_layout();fig.savefig('budget_curve.png',dpi=180)
