"""The frozen research view must work without invoking analysis services."""
import pytest
pytest.importorskip("streamlit", reason="install .[ui] for research UI tests")
from streamlit.testing.v1 import AppTest
from dsa.agent import workbench

def page():
    from dsa.agent.workbench import main
    main()

def research_page(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("research view invoked database, graph, or provider")
    for name in ("_ensure", "mode_banner", "model_panel", "run_evidence_panel"):
        monkeypatch.setattr(workbench, name, forbidden)
    from dsa import config
    monkeypatch.setattr(config, "get_engine", forbidden)
    at = AppTest.from_function(page, default_timeout=30)
    at.session_state["workbench_view"] = "研究证据与业务决策"
    return at.run()

def test_research_view_runs_without_database_or_llm(monkeypatch):
    at = research_page(monkeypatch)
    assert not at.exception
    assert any("不投放" in m.value for m in at.warning)
    assert not any(x.label == "开始分析" for x in at.button)
    assert len(at.dataframe) >= 2

def test_mandatory_quota_and_study_switch_remain_read_only(monkeypatch):
    at = research_page(monkeypatch)
    at.radio(key="research_budget_mode").set_value("强制配额").run()
    assert not at.exception
    assert any("升级评审" in m.value for m in at.warning)
    at.selectbox(key="research_study").set_value("retailhero").run()
    assert not at.exception
    assert any("随机" in m.value for m in at.info)
    assert not at.number_input
