"""Tests for the orchestrator."""

import pytest
from orchestrator.main import Orchestrator, Gateway


def test_orchestrator_exists():
    """Test that Orchestrator can be instantiated."""
    orch = Orchestrator(config={'log_level': 'INFO'})
    assert orch is not None
