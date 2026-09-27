"""Quantitative plot from saved paired.csv; no generated measurements."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/development/paired.csv'));p.add_argument('--out',type=Path,default=Path('paired_deltas.png'));a=p.parse_args()
with a.input.open() as f:rows=list(csv.DictReader(f))
v=sorted(float(x['delta_ndcg10']) for x in rows)
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
axs[0].scatter([float(x['bm25_ndcg10']) for x in rows],[float(x['dense_ndcg10']) for x in rows],s=12,alpha=.22,color='#147d92');axs[0].plot([0,1],[0,1],color='#999999',lw=1);axs[0].set(xlabel='BM25 nDCG@10',ylabel='MiniLM nDCG@10',title=f'Paired queries (n={len(rows)})')
axs[1].bar(range(len(v)),v,width=1,color=['#b66726' if x<0 else '#147d92' for x in v]);axs[1].axhline(0,color='#555555',lw=.7);axs[1].set(xlabel='Queries sorted by difference',ylabel='MiniLM - BM25',title='Per-query nDCG@10 difference')
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180);print(a.out)
