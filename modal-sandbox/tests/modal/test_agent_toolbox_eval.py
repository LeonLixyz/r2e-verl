from relace_agent.modal.agent_toolbox_eval import EvalConfig


def test_load_r2e() -> None:
    config = EvalConfig.load_r2e()
    assert config is not None
    assert config.sandbox_agents
    assert config.ui_build is None
    assert config.ui_test is None
    assert config.ui_deploy is None


def test_load_ui() -> None:
    config = EvalConfig.load_ui()
    assert config is not None
    assert config.sandbox_agents
    assert config.ui_build is not None
    assert config.ui_test is not None
    assert config.ui_deploy is not None


def test_ui_parameters() -> None:
    config = EvalConfig.load_ui()
    for matrix_key in config.matrix:
        params = config.get_parameters(matrix_key)
        assert params.overrides
        assert params.overrides.get("tools")
