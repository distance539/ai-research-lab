"""Quantitative figure only from saved real run diagnostics; no generated numbers."""
import argparse
from pathlib import Path
from common import read

def main(out,dest):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 s=read(out/'summary.json');rows=read(out/'diagnostics.json');fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 names=['Original RRF','Cross-encoder','Candidate oracle'];vals=[s[k]['ndcg_cut_10'] for k in ['original','reranked','oracle']]
 ax[0].bar(names,vals,color=['#389ac1','#4b9970','#dca34a']);ax[0].set_ylim(0,1.08);ax[0].set_ylabel('Mean nDCG@10');ax[0].set_title(f"{s['queries']} development queries, same 50 candidates")
 for i,v in enumerate(vals):ax[0].text(i,v+.02,f'{v:.4f}',ha='center')
 d=sorted(x['delta'] for x in rows);ax[1].bar(range(len(d)),d,color=['#c16b50' if x<0 else '#389ac1' for x in d]);ax[1].axhline(0,color='#444',lw=.7);ax[1].set_xlabel('Queries sorted by paired difference');ax[1].set_ylabel('Reranked minus original nDCG@10');ax[1].set_title('All paired outcomes, no uncertainty interval')
 dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=180);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();main(a.out,a.dest)
