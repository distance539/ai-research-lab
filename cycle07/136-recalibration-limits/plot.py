import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('results.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11,4))
for n in ['raw','source','target']:
 v=r[n];ax[0].plot(range(1,1001),[e/k for k,e in enumerate(v['prefix_errors'],1)],label=n)
ax[0].set(xlabel='Accepted count',ylabel='Observed risk');ax[0].legend();ax[1].bar(['raw','source','target'],[r[n]['nll'] for n in ['raw','source','target']]);ax[1].set(ylabel='Exploration NLL');fig.tight_layout();fig.savefig('repair_limits.png',dpi=180)
