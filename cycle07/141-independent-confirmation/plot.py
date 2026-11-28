from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;p=json.loads((r/'confirmation_predictions.json').read_text());x=json.loads((r/'confirmation_results.json').read_text())
f,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for n,c in [('global','#237caa'),('group','#d58523')]:axs[0].plot(range(1,1001),p[n]['prefix'],label=n,color=c)
axs[0].set(xlabel='Accepted reviews',ylabel='Error fraction',title='Confirmation risk / coverage');axs[0].legend()
a=x['delta_nll'];lo,hi=x['ci95'];axs[1].errorbar(a,0,xerr=[[a-lo],[hi-a]],fmt='o',capsize=6,color='#237caa');axs[1].axvline(0,color='gray',linestyle='--');axs[1].set(yticks=[],xlabel='NLL(group) - NLL(global)',title='Paired bootstrap 95% interval')
(r.parent/'images').mkdir(exist_ok=True)
f.savefig(r.parent/'images/confirmation_results.png',dpi=160)
