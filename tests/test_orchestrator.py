"""Tests for the orchestrator."""


from orchestrator.main import Orchestrator


def test_orchestrator_exists():
    """Test that Orchestrator can be instantiated."""
    orch = Orchestrator(config={"log_level": "INFO"})
    assert orch is not None
