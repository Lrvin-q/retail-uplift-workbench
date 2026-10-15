from importlib.resources import files
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def test_wheel_sql_and_human_readable_sql_have_identical_content():
    packaged=files("dsa.resources").joinpath("sql")
    names={p.name for p in packaged.iterdir() if p.name.endswith(".sql")}
    source_names={p.name for p in (ROOT/"sql").glob("*.sql")}
    assert names==source_names
    for name in names:
        assert packaged.joinpath(name).read_bytes()==(ROOT/"sql"/name).read_bytes()
    assert files("dsa.resources").joinpath("metrics.yaml").read_bytes()==(ROOT/"docs/protocols/metrics.yaml").read_bytes()

def test_packaged_sql_never_sets_a_fixed_database_password():
    packaged=files("dsa.resources").joinpath("sql")
    for p in packaged.iterdir():
        if p.name.endswith(".sql"):
            assert not re.search(r"(?i)password\s+'[^']+'",p.read_text(encoding="utf-8")),p.name
