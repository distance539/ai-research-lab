import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;r=json.loads((p/'results.json').read_text());e=json.loads((p/'formal_env.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(12,4));x=list(r);axs[0].bar(x,[r[k]['flips'] for k in x],color=['#cf8833','#268b9e','#587468']);axs[0].set(ylabel='Prediction flips vs 512 / 1000',xlabel='Token budget');axs[1].bar(x,[v['seconds'] for v in e['conditions']],color='#268b9e');axs[1].set(ylabel='Measured loop seconds (one run)',xlabel='Token budget');fig.tight_layout();fig.savefig(sys.argv[1],dpi=160)
