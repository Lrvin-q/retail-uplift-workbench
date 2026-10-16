"""Published references must match exact original aggregate bytes."""
import hashlib
from pathlib import Path
from dsa.research_portal import load_catalog,load_exploration_ledger

def refs(value):
    if isinstance(value,dict):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from refs(child)
    elif isinstance(value,list):
        for child in value:
            yield from refs(child)

def test_every_catalog_and_ledger_reference_exists_and_matches():
    root=Path(__file__).resolve().parents[1]
    found=list(refs(load_catalog()))+list(refs(load_exploration_ledger()))
    assert len(found)>=16
    for ref in found:
        path=root/ref["path"]
        assert path.is_file(),ref["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref["sha256"]
