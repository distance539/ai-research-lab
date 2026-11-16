"""Hash-checked CPU replay of the historical SST2 protocol audit pool."""
import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from baseline import digest, prepare, save

ROOT = Path(__file__).resolve().parent

def load(name):
    return json.loads((ROOT / name).read_bytes())

def classify(row, logits, tokens):
    maximum = max(logits)
    log_total = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
    probabilities = [math.exp(x - log_total) for x in logits]
    prediction = int(probabilities[1] > probabilities[0])
    return dict(idx=row['idx'], label=row['label'],
                text_sha256=hashlib.sha256(row['sentence'].encode()).hexdigest(),
                logits=logits, retained_tokens=tokens, probabilities=probabilities,
                prediction=prediction, confidence=max(probabilities),
                correct=int(prediction == row['label']), nll=log_total-logits[row['label']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    cfg = load('config.json')
    if args.action == 'prepare':
        prepare(args.cache, cfg)
        return

    import torch
    import transformers
    import numpy
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    start = time.perf_counter()
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    lock = load('resources.lock.json')
    model_dir = args.cache / cfg['model_revision']
    dataset = args.cache / 'validation.parquet'
    assert digest(dataset) == lock['dataset']['sha256']
    for item in lock['files']:
        assert digest(model_dir / item['name']) == item['sha256']
    rows = {r['idx']: r for r in pq.read_table(dataset).to_pylist()}
    ids = load('split_manifest.json')['protocol_audit']
    if args.action == 'smoke':
        ids = ids[:8]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, attn_implementation=cfg['attention']).eval()
    assert model.config.id2label == {0:'NEGATIVE', 1:'POSITIVE'}
    predictions, shapes = [], []
    inference_start = time.perf_counter()
    for offset in range(0, len(ids), cfg['batch_size']):
        batch = [rows[i] for i in ids[offset:offset+cfg['batch_size']]]
        inputs = tokenizer([r['sentence'] for r in batch], padding=True, truncation=True,
                           max_length=cfg['max_length'], return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits
        assert list(logits.shape) == [len(batch), 2] and torch.isfinite(logits).all()
        shapes.append(dict(input=list(inputs['input_ids'].shape), logits=list(logits.shape)))
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        predictions.extend(classify(r, z, t) for r, z, t in zip(batch, logits.tolist(), lengths))
    timings = {args.action: time.perf_counter()-inference_start}
    archived = load('audit_predictions.json')[:len(ids)]
    assert len(archived) == len(predictions)
    for old, new in zip(archived, predictions):
        assert all(old[k]==new[k] for k in ('idx','label','text_sha256','prediction','correct'))
    error = max(abs(x-y) for old,new in zip(archived,predictions)
                for x,y in zip(old['logits'],new['logits']))
    assert error < 1e-5
    save(ROOT / (args.action+'_predictions.json'), predictions)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    timings['total_after_imports'] = time.perf_counter()-start
    assert rss <= cfg['rss_budget_bytes'] and timings['total_after_imports'] < cfg['formal_budget_seconds']
    save(ROOT / (args.action+'_environment.json'), dict(python=platform.python_version(),
         platform=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
         numpy=numpy.__version__, device='cpu', dtype='float32', timings=timings, peak_rss_bytes=rss,
         shapes=shapes, max_logit_difference=error))
    print('replayed',len(ids),'max_logit_difference',error,timings,'peak_RSS_bytes',rss)

if __name__ == '__main__':
    main()
