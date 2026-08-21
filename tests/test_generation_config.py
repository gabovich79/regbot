from services.claude_service import build_generation_config


def test_generation_config_disables_google_search_by_default():
    config = build_generation_config("ענה בעברית", enable_google_search=False)

    assert not config.tools


def test_generation_config_adds_google_search_only_when_explicitly_enabled():
    config = build_generation_config("ענה בעברית", enable_google_search=True)

    assert len(config.tools) == 1
