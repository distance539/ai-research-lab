"""Verify a precomputed immutable release manifest; never silently renew it."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def verify(root=R):
 lock=json.loads((root/'final_lock.json').read_text())
 for name,digest in lock['files'].items():
  path=root/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   raise ValueError('Frozen file mismatch: '+name)
 return lock
if __name__=='__main__':
 verify();print('Frozen files verified')
