"""Measured figure: source is saved summary/per-query JSON, never image generation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main(source,out):
    summary=json.loads((source/'summary.json').read_text())['methods'];per=json.loads((source/'per_query.json').read_text())
    names=['bm25','dense','rrf60','alternate100','rrf60_depth50','alternate50']
    labels=['BM25','MiniLM','RRF 100+100','Alternate 100+100','RRF 50+50','Alternate 50+50']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    ax=axes[0];bars=ax.barh(labels,[summary[n]['ndcg_cut_10'] for n in names],color=['#a56226','#229bb2','#147d65','#62769c','#51a88a','#8a98b3']);ax.invert_yaxis();ax.set_xlim(0,0.82);ax.set_xlabel('Development nDCG@10 (809 queries)')
    ax.bar_label(bars,fmt='%.5f',padding=3,fontsize=8)
    ds=sorted(per['rrf60'][q]['ndcg_cut_10']-per['alternate100'][q]['ndcg_cut_10'] for q in per['rrf60'])
    axes[1].bar(range(len(ds)),ds,width=1,color=['#ad6729' if d<0 else '#168191' for d in ds]);axes[1].axhline(0,color='#444',lw=.8);axes[1].set_xlabel('Queries sorted by paired difference');axes[1].set_ylabel('RRF minus Alternate nDCG@10');axes[1].set_title('Same input union, different ranking rules')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.input,a.out)
