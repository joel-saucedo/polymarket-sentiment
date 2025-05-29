"""Basic test to ensure CI passes for scaffolding chunk."""

import pytest


def test_basic_import():
    """Test that we can import the main package."""
    import polymkt_sent
    assert polymkt_sent is not None


def test_config_files_exist():
    """Test that configuration files exist."""
    import os
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    
    assert (project_root / "config" / "accounts.yml").exists()
    assert (project_root / "config" / "scraper.yml").exists()
    assert (project_root / "config" / "logging.yml").exists()


def test_directory_structure():
    """Test that required directories exist."""
    import os
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    
    assert (project_root / "polymkt_sent").exists()
    assert (project_root / "config").exists()
    assert (project_root / "scripts").exists()
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "http").exists()
    assert (project_root / "tests" / "int").exists()
