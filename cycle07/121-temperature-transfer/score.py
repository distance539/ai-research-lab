"""Score the two fixed single-token answers using one full-vocabulary forward."""
import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
import pyarrow.parquet as pq
import torch
import transformers

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    cfg = json.loads(Path('protocol.json').read_text())
    for name, expected in cfg['input_sha256'].items():
        assert digest(name) == expected
    lock = json.loads(Path('model.lock.json').read_text())
    for item in lock['files']:
        assert digest(args.model_dir / item['name']) == item['sha256']
    assert digest(args.data) == json.loads(Path('resources.lock.json').read_text())['dataset']['sha256']
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    tok = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    data = {r['idx']: r for r in pq.read_table(args.data).to_pylist()}
    rows = []
    for split in ('calibration','audit'):
        base = json.loads(Path(split+'_predictions.json').read_text())
        if args.smoke:
            base = base[:4]
        for b in base:
            row = data[b['idx']]
            assert row['label']==b['label'] and hashlib.sha256(row['sentence'].encode()).hexdigest()==b['text_sha256']
            for source, template in cfg['prompts'].items():
                prompt = template.format(sentence=row['sentence'])
                ctx = tok.encode(prompt, add_special_tokens=False)
                tails = []
                for candidate in cfg['candidates']:
                    full = tok.encode(prompt+candidate, add_special_tokens=False)
                    assert full[:-1]==ctx and len(full)<=cfg['max_context']
                    tails.append(full[-1])
                assert len(set(tails))==2
                with torch.inference_mode():
                    logits = model(torch.tensor([ctx])).logits
                    scores = logits[0,-1].double().log_softmax(0)[tails].tolist()
                rows.append(dict(idx=b['idx'],split=split,source=source,label=b['label'],tokens=len(ctx),
                                 shape=list(logits.shape),candidate_ids=tails,scores=scores))
                assert time.monotonic()-start < 1800
    tag = 'smoke' if args.smoke else 'formal'
    Path(tag+'.json').write_text(json.dumps(rows)+'\n')
    print(json.dumps(dict(rows=len(rows),seconds=time.monotonic()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        os=platform.platform(),python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
        protocol=digest('protocol.json'),code=digest(__file__),result=digest(tag+'.json'))))
if __name__=='__main__':
    main()
