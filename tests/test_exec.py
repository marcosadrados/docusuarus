"""Tests for the execution layer."""

import pytest
from exec.broker import Broker


def test_broker_class_exists():
    """Test that Broker class can be instantiated."""
    broker = Broker(paper_trading=True)
    assert broker is not None
