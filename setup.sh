#!/usr/bin/env bash
# Setup script for tscv-vision
# Purpose: provision a clean local dev env the same way the coding agent/CI would.
# Usage:
#   bash setup.sh [--python 3.11] [--install-only] [--skip-tests] [--skip-lint]
#                  [--no-venv] [--quiet]
# Notes:
#   - Works on macOS/Linux and Windows (Git Bash). Requires bash and curl.

set -Eeuo pipefail
IFS=$'
	'

# ---------------------------
# Config (can be overridden via env or flags)
# ---------------------------
PY_VERSION_DEFAULT="3.11"
PY_VERSION="${PY_VERSION:-$PY_VERSION_DEFAULT}"
QUIET="false"
DO_LINT="true"
DO_TESTS="true"
USE_VENV="true"
INSTALL_ONLY="false"

# ---------------------------
# UX helpers
# ---------------------------
log()  { [[ "$QUIET" == "true" ]] || echo -e "[setup] $*"; }
die()  { echo -e "[setup:ERROR] $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found"; }

# ---------------------------
# Arg parsing
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY_VERSION="$2"; shift 2;;
    --install-only) INSTALL_ONLY="true"; shift;;
    --skip-tests) DO_TESTS="false"; shift;;
    --skip-lint) DO_LINT="false"; shift;;
    --no-venv) USE_VENV="false"; shift;;
    --quiet|-q) QUIET="true"; shift;;
    -h|--help)
      sed -n '1,35p' "$0"; exit 0;;
    *) die "Unknown flag: $1";;
  esac
done

# ---------------------------
# Ensure Python (prefer exact minor)
# ---------------------------
choose_python() {
  local want="$1"
  # Prefer pyenv if available
  if command -v pyenv >/dev/null 2>&1; then
    if pyenv versions --bare | grep -qx "$want"; then
      PY_BIN="$(pyenv root)/versions/${want}/bin/python"
      log "Using Python via pyenv: ${PY_BIN}"
      echo "$PY_BIN"; return 0
    fi
  fi
  # Try system binaries
  for cand in \
    "python${want}" "python${want%.*}" \
    "/usr/bin/python${want}" "/usr/local/bin/python${want}" \
    python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      local ver; ver="$($cand -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
      if [[ "$ver" == "$want" ]]; then echo "$cand"; return 0; fi
      # accept close matches if exact not found
      if [[ -z "${PY_BIN:-}" ]]; then PY_BIN="$cand"; fi
    fi
  done
  # Fallback to any python3
  if command -v python3 >/dev/null 2>&1; then echo python3; return 0; fi
  die "No suitable Python found. Install Python ${want} or set PY_VERSION/PY_BIN."
}

PY_BIN="${PY_BIN:-$(choose_python "$PY_VERSION")}"
log "Python resolved to: $("$PY_BIN" --version 2>&1)"

# ---------------------------
# Ensure Poetry
# ---------------------------
ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    log "Poetry found: $(poetry --version)"; return 0
  fi
  need curl
  log "Installing Poetry (user-local)…"
  curl -sSL https://install.python-poetry.org | "$PY_BIN" -
  # shellcheck disable=SC1090
  if [[ -f "$HOME/.poetry/env" ]]; then source "$HOME/.poetry/env"; fi
  command -v poetry >/dev/null 2>&1 || die "Poetry not available after install"
}
ensure_poetry

# ---------------------------
# Create/use virtualenv
# ---------------------------
if [[ "$USE_VENV" == "true" ]]; then
  log "Configuring Poetry env with ${PY_BIN}"
  poetry env use "$PY_BIN" >/dev/null
fi

# ---------------------------
# Install dependencies
# ---------------------------
log "Installing dependencies (Poetry)…"
poetry install --no-interaction --no-ansi

if [[ "$INSTALL_ONLY" == "true" ]]; then
  log "Install-only mode complete."
  exit 0
fi

# ---------------------------
# Lint & type-check
# ---------------------------
if [[ "$DO_LINT" == "true" ]]; then
  log "Running ruff + mypy…"
  poetry run ruff check .
  poetry run mypy src
else
  log "Skipping lint/type-check per flags"
fi

# ---------------------------
# Tests
# ---------------------------
if [[ "$DO_TESTS" == "true" ]]; then
  log "Running tests…"
  poetry run pytest -q
else
  log "Skipping tests per flags"
fi

# ---------------------------
# Smoke CLI (optional, not failing build)
# ---------------------------
python samples/generate.py >/dev/null 2>&1 || true
if [[ -f samples/sine.npy ]]; then
  log "Smoke: extracting features from generated samples/sine.npy"
  set +e
  poetry run tscv-features --encoder gaf --input samples/sine.npy --output /tmp/feats.npz --bins 16
  set -e
  rm samples/sine.npy
fi

log "Setup complete. You can start developing now."
log "Common commands:
  poetry run pytest -q
  poetry run ruff check .
  poetry run mypy src
  poetry run tscv-features --help"
  