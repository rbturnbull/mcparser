#!/usr/bin/env sh
# Build the Sphinx docs into docs/_build/html. Extra arguments go to sphinx-build.
set -eu

cd "$(dirname "$0")"

# The docs import llmparser (API reference, generated CLI page), so build in the
# project environment with the docs dependency group rather than `uv tool run`.
uv run --group docs sphinx-build -E -W -b html "$@" docs docs/_build/html

echo "To view the documentation, run:"
echo "  python -m http.server --directory docs/_build/html"
