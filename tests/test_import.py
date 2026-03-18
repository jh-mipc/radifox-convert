def test_import_radifox_convert():
    import radifox.convert  # noqa: F401
    assert "unknown" not in radifox.convert.__version__
