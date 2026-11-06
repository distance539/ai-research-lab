"""Render all predeclared audit cells from result.json; no inferred uncertainty."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output', default='transfer_nll.png')
args=p.parse_args()
result=json.loads(Path('result.json').read_text())['audit']
sources=['unit','C','A','B']
targets=['C','A','B']
values=[[result[t][s]['nll'] for s in sources] for t in targets]
fig,ax=plt.subplots(figsize=(9,4.7))
im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1.05,aspect='auto')
ax.set(xticks=range(4),xticklabels=['No scaling','Classifier','Prompt A','Prompt B'],
       yticks=range(3),yticklabels=['Classifier','Prompt A','Prompt B'],
       xlabel='Temperature fitted on',ylabel='Scores evaluated on',
       title='Mean audit NLL  |  360 paired reviews per cell')
for i,row in enumerate(values):
    for j,value in enumerate(row):
        ax.text(j,i,f'{value:.4f}',ha='center',va='center',color='white' if value>.65 else '#111827')
fig.colorbar(im,ax=ax,label='NLL (lower is better)')
fig.tight_layout()
fig.savefig(args.output,dpi=200)
