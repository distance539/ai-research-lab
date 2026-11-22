import copy,json
from analyze import validate,read
p=read('pools.json');r=read('exploration.json');caught=[]
for name in ['confirmation_id','duplicate_movie','nan_logit','missing_row']:
 a=copy.deepcopy(p);b=copy.deepcopy(r)
 if name=='confirmation_id':b[0]['id']=a['confirmation'][0]['id']
 if name=='duplicate_movie':a['calibration'][0]['movie']=a['exploration'][0]['movie']
 if name=='nan_logit':b[0]['logits'][0]=float('nan')
 if name=='missing_row':b.pop()
 try:validate(a,b,'exploration')
 except AssertionError:caught.append(name)
assert len(caught)==4
print(json.dumps({'caught':caught,'status':'passed'}))
