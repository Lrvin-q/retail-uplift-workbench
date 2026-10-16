"""The CLI must emit valid UTF-8 even when the host defaults to cp1252."""
import json,os,subprocess,sys
from pathlib import Path
import pytest

@pytest.mark.parametrize("args",[[],["--json"],["--list"],["--ledger"]])
def test_cli_survives_non_unicode_host_console(tmp_path,args):
    env={k:v for k,v in os.environ.items()
         if not any(x in k.upper() for x in ("DSA_","DATABASE_URL","DEEPSEEK","PYTHONPATH"))}
    env.update(PYTHONIOENCODING="cp1252:strict",PYTHONUTF8="0",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
    completed=subprocess.run([sys.executable,"-m","dsa.research_portal",*args],
                             cwd=tmp_path,env=env,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stderr.decode("utf-8",errors="replace")
    text=completed.stdout.decode("utf-8",errors="strict")
    if args:
        data=json.loads(text)
        if args==["--json"]:
            assert data["decision"]["action"]=="do_not_contact"
            assert data["policy"]["role"]=="primary_frozen"
        elif args==["--list"]:
            assert len(data)==4
        else:
            assert data["project_wide_type1_control"] is False
    else:
        assert "不投放" in text
