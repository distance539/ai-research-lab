"""Real BEIR -> Elasticsearch BM25 -> BEIR/trec_eval pipeline, development split only."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import logging
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from common import ROOT, metrics, qrels, ranked, read_json, rows, sha, verify_resources, write_json

def data_audit(cache):
    corpus = rows(cache / 'scifact/corpus.jsonl')
    queries = rows(cache / 'scifact/queries.jsonl')
    docs = {d['_id']: d for d in corpus}
    qs = {q['_id']: q['text'] for q in queries}
    assert len(docs) == len(corpus) == 5183
    assert len(qs) == len(queries), 'duplicate query IDs'
    original = rows(cache / 'downloads/data/corpus.jsonl')
    assert len(original) == len(docs)
    whitespace_changed = []
    for d in original:
        b = docs[str(d['doc_id'])]
        assert b['title'] == d['title']
        original_text = ' '.join(d['abstract'])
        if b['text'] != original_text:
            whitespace_changed.append(str(d['doc_id']))
        assert ' '.join(b['text'].split()) == ' '.join(original_text.split())
    report = {'corpus': len(docs), 'all_queries': len(qs), 'corpus_titles_exact': True,
              'corpus_text_equal_after_whitespace_normalization': True,
              'whitespace_changed_count': len(whitespace_changed), 'whitespace_changed_ids': whitespace_changed}
    sets = []
    for beir, raw in [('train', 'train'), ('test', 'dev')]:
        rel = qrels(cache / f'scifact/qrels/{beir}.tsv')
        claims = rows(cache / f'downloads/data/claims_{raw}.jsonl')
        assert set(rel) == {str(x['id']) for x in claims}
        for x in claims:
            q = str(x['id'])
            assert qs[q] == x['claim']
            assert set(rel[q]) == set(map(str, x['cited_doc_ids']))
            assert set(rel[q]) <= docs.keys()
            assert all(score == 1 for score in rel[q].values())
        report[beir] = {'queries': len(rel), 'qrel_pairs': sum(map(len, rel.values())),
                        'original_split': raw, 'all_qrels_equal_cited_doc_ids': True,
                        'qrels_equal_evidence_count': sum(set(rel[str(x['id'])]) == set(x['evidence']) for x in claims),
                        'empty_evidence_count': sum(not x['evidence'] for x in claims)}
        sets.append(set(rel))
    assert not sets[0] & sets[1]
    assert not set(qs) & set(docs), 'official identical-ID filter would change the task'
    report['query_doc_id_collision'] = 0
    report['train_test_query_overlap'] = 0
    report['test_usage'] = 'structure and mapping only; no retrieval, error inspection or tuning'
    return report

def main(args):
    start = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    state = {'stages': {}, 'command': [sys.executable, *sys.argv], 'exit_code': None}
    write_json(out / 'status.json', state)
    try:
        config = read_json(ROOT / 'config.json')
        assert config['split'] == 'train', 'Confirmation split is deliberately not an option.'
        lock, upstream = verify_resources(args.cache)
        state['stages']['download'] = {'status': 'passed', 'mode': 'reused verified cache'}
        t = time.perf_counter()
        audit = data_audit(args.cache)
        write_json(out / 'data_audit.json', audit)
        state['stages']['preprocessing'] = {'status': 'passed', 'seconds': time.perf_counter() - t}
        sys.path.insert(0, str(upstream.resolve()))
        from beir.datasets.data_loader import GenericDataLoader
        from compat import RunnableBM25 as BM25Search
        from beir.retrieval.evaluation import EvaluateRetrieval
        import pytrec_eval
        corpus, queries, gold = GenericDataLoader(str(args.cache / 'scifact')).load(split='train')
        if args.limit:
            ids = sorted(queries, key=int)[:args.limit]
            queries = {q: queries[q] for q in ids}
            gold = {q: gold[q] for q in ids}
        code_hashes = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
        identity = {'config': config, 'code': code_hashes, 'data': lock['data_files'],
                    'upstream': lock['beir_commit'], 'model_revision': config['model_revision'],
                    'tokenizer': config['tokenizer'], 'prompt': None, 'decoding': None,
                    'query_ids': list(queries), 'requirements': sha(ROOT / 'requirements.txt')}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        index = 'cycle6-082-' + key[:12]
        write_json(out / 'identity.json', {'cache_key': key, **identity})
        model = BM25Search(index_name=index, hostname=args.hostname, initialize=False,
                           number_of_shards=1, language='english', batch_size=64)
        es = model.es.es
        info = es.info()
        assert info['version']['number'] == config['es_version']
        assert info['version']['build_hash'] == config['model_revision']
        write_json(out / 'es_info.json', info)
        t = time.perf_counter()
        # Fresh index only: upstream initialise() deletes indices, so deliberately never call it.
        exists = es.indices.exists(index=index)
        if exists:
            assert args.reuse_index, 'Index already exists; use --reuse-index with this content-addressed identity.'
            mapping = es.indices.get_mapping(index=index)[index]['mappings']['properties']
            assert mapping['title']['analyzer'] == mapping['txt']['analyzer'] == 'english'
            # Verify all indexed text before reuse, rather than trusting the index name alone.
            from elasticsearch.helpers import scan
            stored = {x['_id']: x['_source'] for x in scan(es, index=index)}
            assert set(stored) == set(corpus)
            for d in corpus:
                assert stored[d]['title'] == corpus[d]['title'] and stored[d]['txt'] == corpus[d]['text']
        else:
            model.es.create_index()
            model.index(corpus)
        es.indices.refresh(index=index)
        assert es.count(index=index)['count'] == len(corpus)
        state['stages']['indexing'] = {'status': 'passed', 'seconds': time.perf_counter() - t, 'documents': len(corpus), 'reused': exists}
        write_json(out / 'index_settings.json', es.indices.get(index=index))
        write_json(out / 'analyzer_example.json', es.indices.analyze(index=index, body={'analyzer': 'english', 'text': queries[next(iter(queries))]}))
        t = time.perf_counter()
        retriever = EvaluateRetrieval(model, k_values=config['k_values'])
        raw = retriever.retrieve(corpus, queries)
        state['stages']['retrieval'] = {'status': 'passed', 'seconds': time.perf_counter() - t}
        write_json(out / 'raw_results.json', raw)
        assert set(raw) == set(queries), 'Missing queries must not disappear from the denominator.'
        results = {q: {d: raw[q][d] for d in ranked(raw[q])[:config['top_k']]} for q in queries}
        write_json(out / 'results.json', results)
        write_json(out / 'qrels_used.json', gold)
        with (out / 'run.trec').open('w') as f:
            for q, values in results.items():
                for rank, d in enumerate(ranked(values), 1):
                    f.write(f'{q} Q0 {d} {rank} {values[d]:.10g} cycle6-082\n')
        t = time.perf_counter()
        official = EvaluateRetrieval.evaluate(gold, results, config['k_values'])
        evaluator = pytrec_eval.RelevanceEvaluator(gold, {'ndcg_cut.1,10,100', 'recall.1,10,100'})
        per_query = evaluator.evaluate(results)
        assert set(per_query) == set(queries)
        worst = 0.0
        diagnoses = []
        for q in queries:
            order = ranked(results[q])
            independent = metrics(gold[q], order)
            for name, value in per_query[q].items():
                worst = max(worst, abs(value - independent[name]))
            per_query[q]['mrr_10'] = independent['mrr_10']
            ranks = {d: order.index(d) + 1 if d in order else None for d in gold[q]}
            diagnoses.append({'query_id': q, 'relevant_ranks': ranks, 'no_relevant_top100': all(x is None for x in ranks.values()),
                              'no_relevant_top10': all(x is None or x > 10 for x in ranks.values()),
                              'raw_count': len(raw[q]), 'kept_count': len(order)})
        assert worst < 1e-10, worst
        write_json(out / 'per_query.json', per_query)
        write_json(out / 'diagnostics.json', diagnoses)
        summary = {name: sum(x[name] for x in per_query.values()) / len(queries) for name in next(iter(per_query.values()))}
        write_json(out / 'summary.json', {'n_queries': len(queries), 'n_documents': len(corpus), 'scope': 'development',
                    'metrics': summary, 'official_beir_rounded': official, 'independent_max_error': worst,
                    'missing_top100': sum(x['no_relevant_top100'] for x in diagnoses),
                    'missing_top10': sum(x['no_relevant_top10'] for x in diagnoses),
                    'raw_count_histogram': dict(collections.Counter(map(len, raw.values())))})
        state['stages']['evaluation'] = {'status': 'passed', 'seconds': time.perf_counter() - t}
        q = next(iter(queries)); d = ranked(results[q])[0]
        write_json(out / 'explanation.json', es.explain(index=index, id=d, body={'query': {'multi_match': {'query': queries[q], 'type': 'best_fields', 'fields': ['title', 'txt'], 'tie_breaker': 0.5}}}))
        stats = es.nodes.stats(metric='jvm,process')
        write_json(out / 'server_resource_snapshot.json', stats)
        packages = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'elasticsearch', 'pytrec-eval-terrier', 'tqdm', 'urllib3', 'certifi']}
        write_json(out / 'environment.json', {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                   'device': 'CPU', 'framework': 'BEIR + Elasticsearch/Lucene', 'packages': packages,
                   'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                   'server_peak_rss': 'not measured; JVM heap snapshot saved separately', 'gpu_memory': 'not applicable'})
        state['exit_code'] = 0
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        state['exit_code'] = 1
        state['error'] = traceback.format_exc()
        raise
    finally:
        state['elapsed_seconds'] = time.perf_counter() - start
        write_json(out / 'status.json', state)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hostname', default='http://127.0.0.1:19282')
    ap.add_argument('--limit', type=int, default=0, help='smoke only; 0 means all development queries')
    ap.add_argument('--reuse-index', action='store_true', help='verify all stored text then reuse the content-addressed experiment index')
    logging.basicConfig(level=logging.WARNING)
    main(ap.parse_args())
