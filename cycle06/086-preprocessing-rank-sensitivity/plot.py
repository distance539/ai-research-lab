"""Plot saved actual measurements only."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();s=json.loads((a.out/'summary.json').read_text());names=['title_256','title_512','abstract_256','abstract_512'];colors=['#1976a5','#4cacc5','#d78a32','#e8b66b']
fig,ax=plt.subplots(1,2,figsize=(12,4.5));base=s['metrics']['rrf']['ndcg_cut_10'];vals=[s['metrics'][n]['ndcg_cut_10'] for n in names]
ax[0].scatter(range(4),vals,c=colors,s=90,zorder=3);ax[0].set_xlim(-.5,3.5);ax[0].axhline(base,color='#38424e',ls='--',label=f'Frozen RRF {base:.4f}');ax[0].set_ylim(.55,.75);ax[0].set_ylabel('nDCG@10 (development)');ax[0].legend(loc='upper right',fontsize=8)
for i,v in enumerate(vals):ax[0].text(i,v+.003,f'{v:.4f}',ha='center',fontsize=9)
x=[s['tokens'][n]['relevant_truncated'] for n in names];ax[1].bar(range(4),x,color=colors);ax[1].set_ylabel('Truncated relevant query-document pairs');ax[1].set_ylim(0,120)
for i,v in enumerate(x):ax[1].text(i,v+2,str(v),ha='center')
for a1 in ax:a1.set_xticks(range(4),[n.replace('_','\n') for n in names]);a1.spines[['top','right']].set_visible(False)
fig.suptitle('100 development queries | same 50 candidates | frozen MiniLM cross-encoder');fig.tight_layout();fig.savefig(a.dest,dpi=180);print(a.dest)
