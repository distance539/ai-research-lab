"""Narrow adapter for BEIR ef83d29's abstract dense-only methods.

All indexing/search/scoring code remains upstream and byte-unmodified.
Dense embedding APIs are deliberately rejected for a lexical retriever.
"""
from beir.retrieval.search.lexical import BM25Search

class RunnableBM25(BM25Search):
    def encode(self, *args, **kwargs):
        raise TypeError('BM25 has no dense embeddings. Use index(corpus).')

    def search_from_files(self, *args, **kwargs):
        raise TypeError('BM25 cannot read dense embeddings. Use search(corpus, queries, top_k).')
