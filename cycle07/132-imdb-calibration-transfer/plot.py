import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
v=json.loads(Path(__file__).with_name('results.json').read_text())
fig,ax=plt.subplots(figsize=(8,4.3))
for i,(key,color) in enumerate([('raw','#39778c'),('frozen','#c18244')]):
 bars=ax.bar([x+(i-.5)*.30 for x in (0,1)],[v[d][key]['nll'] for d in ('SST2','IMDb')],width=.30,label=key,color=color);ax.bar_label(bars,fmt='%.3f',padding=3)
ax.set_xticks([0,1],['SST-2 audit (360)','IMDb exploration (1,000)']);ax.set_ylabel('Mean negative log-likelihood');ax.set_ylim(0,ax.get_ylim()[1]*1.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(sys.argv[1],dpi=200)
