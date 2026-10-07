"""Independent label-balanced epoch scheduling and numerical checkpoint identity."""
import random,hashlib

def schedule(c,target,smoke):
    result={}
    for name,spec in c['arms'].items():
        rng=random.Random(spec['seed']);batches=[]
        for epoch in range(4):
            groups=[[r for r in target if r['gold']==g] for g in (0,1)]
            if spec['mode']=='shuffle':
                for group in groups:rng.shuffle(group)
            for neg,pos in zip(*groups):batches.append(('target',[neg,pos]))
        result[name]=batches[:2] if smoke else batches
    return result

def state_hash(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        h.update(n.encode());h.update(str(list(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
