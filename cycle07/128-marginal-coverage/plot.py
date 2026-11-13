"""Render only the observed partition counts; no smoothing or uncertainty interval."""
import argparse,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=json.loads(Path(__file__).with_name('splits.json').read_text())
    fig,axs=plt.subplots(2,1,figsize=(10,5.8),sharex=True,layout='constrained')
    for ax,counts,n,title,color in zip(axs,([r['hits'] for r in rows],[360-len(r['audit_missed']) for r in rows]),(128,360),('Held-out half (128 per split)','Fixed audit set (360 per split)'),('#187789','#ae6224')):
        h=Counter(counts);ax.bar([100*x/n for x in h],h.values(),width=75/n,color=color);ax.axvline(90,color='#393939',ls='--');ax.set_title(title,loc='left');ax.set_ylabel('Splits');ax.spines[['top','right']].set_visible(False)
    axs[-1].set_xlabel('Empirical coverage (%)');fig.savefig(args.output,dpi=180)
