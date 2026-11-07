"""Render saved risk/coverage measurements; empty sets omitted."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='risk_curve.png')
args = parser.parse_args()
curve = json.loads(Path(__file__).with_name('curve.json').read_bytes())
fig, ax = plt.subplots(figsize=(10, 4.6))
for name, label, color in [('random','ID shuffle','#d47e26'), ('ranked','Confidence order','#127e83')]:
    pts = curve[name][1:]
    ax.plot([p['coverage']*100 for p in pts], [p['risk']*100 for p in pts],
            label=label, color=color, linewidth=1.6)
ax.axhline(curve['uniform_expected_risk']*100, ls='--', color='#62626b', label='Uniform expectation')
ax.set(xlabel='Accepted samples (% of 360)', ylabel='Errors among accepted (%)', xlim=(0,100))
ax.legend(frameon=False)
ax.grid(alpha=.17)
fig.tight_layout()
fig.savefig(args.output, dpi=180)
