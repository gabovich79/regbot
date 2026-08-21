from services.claude_service import build_generation_config


def test_generation_config_disables_google_search_by_default():
    config = build_generation_config("ענה בעברית", enable_google_search=False)

    assert not config.tools


def test_generation_config_never_enables_google_search_tool():
    config = build_generation_config("ענה בעברית", enable_google_search=True)

    assert not config.tools
