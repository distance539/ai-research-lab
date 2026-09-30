import time
from common import write_json
class Encoder:
    """Independent single-process BEIR adapter; official encode executes the model."""
    def __init__(self, model, out, ids):
        self.model, self.out, self.ids = model, out, ids
        self.trace = {}
    def encode(self, texts, role, **kwargs):
        import numpy as np
        t=time.perf_counter()
        lengths=[len(x) for x in self.model.tokenizer(texts, truncation=False, verbose=False)['input_ids']]
        self.trace[role]={'count':len(texts),'tokens_untruncated':sum(lengths),'tokens_after_truncation':sum(min(x,256) for x in lengths),'truncated_count':sum(x>256 for x in lengths),'max_tokens':max(lengths)}
        arr=self.model.encode(texts, **kwargs)
        assert arr.shape==(len(texts),384) and arr.isfinite().all()
        assert (arr.norm(dim=1)-1).abs().max()<1e-5
        self.trace[role].update(seconds=time.perf_counter()-t,shape=list(arr.shape),max_norm_error=float((arr.norm(dim=1)-1).abs().max()))
        # Embeddings stay in the explicit external cache, excluded from distribution.
        np.save(self.embedding_cache/(role+'.npy'),arr.cpu().numpy())
        write_json(self.out/(role+'_lengths.json'),dict(zip(self.ids[role],lengths)))
        write_json(self.out/'encoding.json', self.trace)
        return arr
    def encode_queries(self, queries, **kwargs):return self.encode(queries,'queries',**kwargs)
    def encode_corpus(self, corpus, **kwargs):return self.encode([(x.get('title','')+' '+x['text']).strip() for x in corpus],'corpus',**kwargs)
