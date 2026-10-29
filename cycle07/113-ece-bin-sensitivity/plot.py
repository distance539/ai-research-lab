import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
a=json.loads((P/'analysis_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
cs=a['conditions'];x=[r['bins'] for r in cs];y=[r['ece'] for r in cs]
axes[0].errorbar(x,y,yerr=[[v-r['bootstrap95'][0] for v,r in zip(y,cs)],[r['bootstrap95'][1]-v for v,r in zip(y,cs)]],fmt='o',capsize=5,color='#087e8b');axes[0].set(xlabel='Equal-width bins',ylabel='ECE',xticks=x,ylim=(0,.14),title='Same 256 predictions; paired bootstrap 95%')
b=cs[-1]['per_bin']; b=[v for v in b if v['n']];xx=[v['confidence'] for v in b];yy=[v['accuracy'] for v in b]
axes[1].errorbar(xx,yy,yerr=[[y-v['accuracy_wilson95'][0] for y,v in zip(yy,b)],[v['accuracy_wilson95'][1]-y for y,v in zip(yy,b)]],fmt='o',capsize=4,color='#dd7f2a');axes[1].plot([.5,1],[.5,1],'--',color='gray')
for v in b:axes[1].annotate('n='+str(v['n']),(v['confidence'],v['accuracy']),xytext=((-14,8) if v['bin']==12 else (12,20) if v['bin']==13 else (0,8)),textcoords='offset points',fontsize=8,ha='center')
axes[1].set(xlim=(.48,1.04),ylim=(-.04,1.12),xlabel='Mean confidence',ylabel='Accuracy (pointwise Wilson 95%)',title='20 bins: empty bins omitted, all counts shown')
fig.savefig(sys.argv[1] if len(sys.argv)>1 else P/'ece_results.png',dpi=160)
print('matplotlib',matplotlib.__version__)
