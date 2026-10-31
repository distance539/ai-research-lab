import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rows=json.loads((p/'group_results.json').read_text())['groups'][1:6]
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.bar([r['group'] for r in rows],[100*r['high_error_rate'] for r in rows],color=['#138f9c']*3+['#d58a32']*2)
for i,r in enumerate(rows):ax.text(i,100*r['high_error_rate']+.4,f"{r['high_errors']}/{r['high_n']}",ha='center')
ax.set_ylim(0,19);ax.set_ylabel('Error rate among confidence >= 0.9 (%)');ax.set_title('Frozen exploratory predictions: length and original-label groups');fig.savefig(sys.argv[1] if len(sys.argv)>1 else 'group_results.png',dpi=180)
