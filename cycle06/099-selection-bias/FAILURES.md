# Failure logs
Original private logs retained outside distribution. Only machine paths anonymized here.

## smoke (exit1, path-anonymized copy)
Traceback (most recent call last):
  File "<workspace>/drafts/research-trends/2026-10-14_099_selection_bias/code/run.py", line 83, in <module>
    try:main(a)
        ^^^^^^^
  File "<workspace>/drafts/research-trends/2026-10-14_099_selection_bias/code/run.py", line 22, in main
    import torch,transformers,pyarrow.parquet as pq,datasets,yaml
  File "<environment>/lib/python3.12/site-packages/pyarrow/__init__.py", line 59, in <module>
    from pyarrow.lib import (BuildInfo, CppBuildInfo, RuntimeInfo, set_timezone_db_path,
  File "pyarrow/lib.pyx", line 42, in init pyarrow.lib
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4

## tests (exit1, path-anonymized copy)
Traceback (most recent call last):
  File "<workspace>/drafts/research-trends/2026-10-14_099_selection_bias/code/test_checks.py", line 5, in <module>
    from select import choose
ImportError: cannot import name 'choose' from 'select' (unknown location)
