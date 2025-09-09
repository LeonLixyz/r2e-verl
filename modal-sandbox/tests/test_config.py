from relace_agent.config import RelaceConfig


def test_relace_config() -> None:
    config = RelaceConfig.load_default()

    assert len(config.user_agents) > 0
