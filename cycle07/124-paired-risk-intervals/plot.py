import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--output', default='gain_intervals.png')
args=parser.parse_args()
rows=json.loads(Path(__file__).with_name('points.json').read_text())[1:]
x=[r['k'] for r in rows]
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
ax.fill_between(x,[100*r['ci'][0] for r in rows],[100*r['ci'][1] for r in rows],
                color='#b9ddd8',label='95% pointwise paired bootstrap')
ax.plot(x,[100*r['gain'] for r in rows],color='#17485b',label='Observed gain')
ax.axhline(0,color='#666',lw=.8)
ax.axvline(270,color='#ae6628',ls=':',label='Primary k = 270')
ax.set(xlabel='Accepted sentences (out of 360)',ylabel='Shuffle risk minus confidence risk (pp)')
ax.legend(frameon=False)
fig.savefig(args.output,dpi=180)
