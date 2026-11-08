"""Plot actual ranked errors; comparison uses every saved prefix."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='prefix_errors.png')
args = parser.parse_args()
data = json.loads(Path(__file__).with_name('ranking.json').read_bytes())
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
for name, marker, offset, color in [('maximum','o',0,'#005c80'),('entropy','s',9,'#d47722'),('gap','^',18,'#197951')]:
    points = data['curves'][name]
    left.step(range(len(points)), [p['errors'] for p in points], where='post',
              label=name, color=color, marker=marker, markevery=(offset,30), markersize=4, linewidth=1)
left.set(xlabel='Accepted examples', ylabel='Errors among accepted examples', title='Three score rules on the same predictions')
left.legend()
baseline = data['orders']['maximum']
for row, name in enumerate(['entropy','gap']):
    mismatches = [int(a != b) for a,b in zip(baseline,data['orders'][name])]
    right.barh(row, sum(mismatches), color='#197951')
    right.text(sum(mismatches)+3,row,f'{sum(mismatches)} / {len(mismatches)}',va='center')
right.set(yticks=[0,1], yticklabels=['Entropy','Absolute gap'], xlim=(0,360),
          xlabel='Positions with a different sample ID',title='Position-by-position vs. max probability')
fig.savefig(args.output, dpi=170)
