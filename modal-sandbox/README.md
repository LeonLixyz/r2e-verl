# modal sandbox

Setup local dev environment using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync
```

Login to modal:

```sh
uv run modal setup
```

Deploy sandbox to modal:

```sh
uv run modal deploy -m relace_agent.modal.agent_toolbox
```
