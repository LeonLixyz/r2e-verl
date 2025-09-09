# relace-agent

[![Tests Workflow](https://github.com/squack-io/relace-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/squack-io/relace-agent/actions/workflows/tests.yml)
[![Deploy Workflow](https://github.com/squack-io/relace-agent/actions/workflows/deploy.yml/badge.svg)](https://github.com/squack-io/relace-agent/actions/workflows/deploy.yml)
[![Modal Deployment](https://img.shields.io/badge/app-grey?logo=modal)](https://modal.com/apps/relace/main/deployed/Relace-Agent)
[![Storage](https://img.shields.io/badge/storage-grey?logo=modal)](https://modal.com/storage/relace/main/relace-agent)
[![Database](https://img.shields.io/badge/database-black?logo=supabase)](https://supabase.com/dashboard/project/phspkymemoiorxsgenpy)
[![FastAPI Docs](https://img.shields.io/badge/docs-blue?logo=fastapi)](https://agent.modal-origin.relace.run/docs)

Setup local dev environment using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync
```

Run lints/tests locally:

```sh
uv run ruff check
uv run mypy .
uv run pytest
```

Deploy to modal:

```sh
uv run modal deploy -m relace_agent.modal.agent_runner
uv run modal deploy -m relace_agent.modal.agent_toolbox
```
