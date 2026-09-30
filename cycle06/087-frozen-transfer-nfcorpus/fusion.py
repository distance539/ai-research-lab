import importlib.util,sys
from common import ROOT
def official_modules():
    # Execute two complete, byte-identical upstream modules without Java/search imports.
    # The upstream fusion import points at this exact official trectools module.
    def load(name, filename):
        spec=importlib.util.spec_from_file_location(name,ROOT/'vendor'/filename)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
    trec=load('pyserini.trectools','trectools_base.py')
    fusion=load('_pinned_pyserini_fusion','fusion_base.py')
    return trec, fusion
