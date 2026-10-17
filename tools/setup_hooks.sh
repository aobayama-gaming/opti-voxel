#!/usr/bin/env bash
# Points git at the repo's hooks (.githooks/). Run once per clone: the setting lives in the shared
# repository config, so every worktree of the clone gets it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$ROOT" config core.hooksPath .githooks
echo ">> hooks enabled: $(git -C "$ROOT" config core.hooksPath)"
