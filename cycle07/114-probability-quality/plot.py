import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path(__file__).with_name('temperature_results.json').read_text());x=[str(v['temperature']) for v in r]
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,key,title in zip(axes,['accuracy','nll','brier'],['Accuracy','Negative log-likelihood','Brier (two-class sum)']):
 ax.plot(x,[v[key] for v in r],'o-',color='#167d9a');ax.set_title(title);ax.set_xlabel('Temperature');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
fig.suptitle('Fixed256 exploratory predictions | no fitting | no uncertainty estimate')
fig.savefig(sys.argv[1],dpi=160)
