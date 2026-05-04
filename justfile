set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

LIB := "my_library"

test-install:
	uv sync --extra test


nbktest-install:
	uv sync --extra notebooks

dev-install:
	uv sync --extra dev

publish-install:
	uv sync --extra build --extra dev

check: lint typecheck test doctest

build: typecheck test
	uv run python -m build

lint:
	uv run ruff check {{LIB}}

format:
	uv run ruff format {{LIB}}

test:
	uv run pytest --disable-warnings

typecheck:
	uv run mypy {{LIB}}/ --config-file pyproject.toml

doctest:
	uv run pytest --doctest-modules {{LIB}}

coverage: 
	uv run pytest --cov-report html --cov={{LIB}} tests/

docs:
	uv run mkdocs build

clean:
	uv run python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
	uv run python -c "import shutil; shutil.rmtree('htmlcov', ignore_errors=True)"
	uv run python -c "import os; os.remove('.coverage') if os.path.exists('.coverage') else None"
	uv run python -c "import shutil; shutil.rmtree('site', ignore_errors=True)"

FORCE: