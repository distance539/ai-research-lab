"""Generate an overlay and a threshold operating point from saved logs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('output')
args=parser.parse_args()
rows=json.loads(Path('paired.json').read_bytes())
summary=json.loads(Path('results.json').read_bytes())
fig, ax=plt.subplots(figsize=(9,4.6),layout='constrained')
for name,color,style in [('raw','#164f75','-'),('scaled','#d98020','--')]:
    curve=rows[name]['curve'][1:]
    ax.plot([r['k']/360 for r in curve],[r['risk'] for r in curve],style,color=color,label=name)
    p=summary[name]['fixed_threshold']
    if p['k']:
        ax.scatter(p['k']/360,p['risk'],color=color,label=name+' at 0.99')
ax.set(xlabel='Coverage (accepted / 360)',ylabel='Errors / accepted',title='SST-2 audit | Risk at matched coverage')
ax.legend()
fig.savefig(args.output,dpi=200)
