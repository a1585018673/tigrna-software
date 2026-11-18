"""Verify all distributed files listed in MANIFEST.sha256 (standard library only)."""
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
count=0
for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    target=(root/name).resolve()
    if root not in target.parents:
        raise ValueError(f'Unsafe manifest path: {name}')
    actual=hashlib.sha256(target.read_bytes()).hexdigest()
    if expected!=actual:
        raise ValueError(f'Hash mismatch: {name}')
    count+=1
print(f'All {count} manifest file checksums match.')
