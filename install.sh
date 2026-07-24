#!/usr/bin/env bash
# trae-agent one-shot installer
# Inspired by https://opencode.ai/install and OpenClaw's install flow.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/1982167424-art/trae-agent/main/install.sh | bash
#   ./install.sh                  # install to ~/.local/bin
#   ./install.sh --system         # install to /usr/local/bin (requires sudo)
#   ./install.sh --uninstall      # remove the installed binary

set -euo pipefail

REPO_URL="https://github.com/1982167424-art/trae-agent.git"
PACKAGE_NAME="trae-agent"
USE_SYSTEM=0
UNINSTALL=0

for arg in "$@"; do
    case "$arg" in
        --system)    USE_SYSTEM=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --help|-h)
            echo "trae-agent installer"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --system      Install to /usr/local/bin instead of ~/.local/bin"
            echo "  --uninstall   Remove the existing installation"
            echo "  --help, -h    Show this help"
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# --- Pick install path ---------------------------------------------------------
if [ "$USE_SYSTEM" -eq 1 ]; then
    INSTALL_BIN="/usr/local/bin/trae"
else
    # XDG-style: ~/.local/bin if it exists or can be created, else fall back.
    INSTALL_BIN="$HOME/.local/bin/trae"
    mkdir -p "$(dirname "$INSTALL_BIN")"
fi

if [ "$UNINSTALL" -eq 1 ]; then
    echo "Uninstalling trae-agent..."
    if [ -e "$INSTALL_BIN" ]; then
        rm -f "$INSTALL_BIN"
        echo "Removed: $INSTALL_BIN"
    else
        echo "Nothing to remove at $INSTALL_BIN"
    fi
    exit 0
fi

# --- Preflight ----------------------------------------------------------------
echo "▸ Checking Python..."
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "✗ Python 3.10+ is required but not installed." >&2
    echo "  Install it from https://www.python.org/" >&2
    exit 1
fi

PY_VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PY" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PY" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "✗ Python 3.10+ required, found $PY_VERSION" >&2
    exit 1
fi
echo "  ✓ Python $PY_VERSION"

# --- Locate or install uv -----------------------------------------------------
echo "▸ Checking uv..."
if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
elif command -v pipx >/dev/null 2>&1; then
    echo "  → uv not found, falling back to pipx"
    UV=""  # use pipx path below
else
    echo "  → uv not found, installing via pip"
    "$PY" -m pip install --user uv >/dev/null 2>&1 || {
        echo "✗ Failed to install uv automatically." >&2
        echo "  Please install it: https://docs.astral.sh/uv/" >&2
        exit 1
    }
    UV="$HOME/.local/bin/uv"
fi
echo "  ✓ uv: $(command -v "$UV" 2>/dev/null || echo 'via pipx/pip')"

# --- Install the package ------------------------------------------------------
echo "▸ Installing trae-agent to $INSTALL_BIN..."

if [ -n "$UV" ]; then
    # uv tool install — sandboxed, isolated env, exposes entry points as `trae`.
    "$UV" tool install --from "git+${REPO_URL}" "$PACKAGE_NAME" 2>&1 | tail -5 || {
        echo "✗ uv tool install failed. Trying alternative..." >&2
        "$UV" pip install --system "trae-agent @ git+${REPO_URL}" 2>&1 | tail -5
    }
    # uv tool puts binaries in ~/.local/bin by default; symlink for convenience.
    TOOL_BIN="$HOME/.local/bin/trae"
    if [ -e "$TOOL_BIN" ] && [ "$INSTALL_BIN" != "$TOOL_BIN" ]; then
        ln -sf "$TOOL_BIN" "$INSTALL_BIN"
    fi
else
    # pipx fallback.
    pipx install "git+${REPO_URL}" 2>&1 | tail -5
fi

# --- Check Ollama (optional) --------------------------------------------------
echo "▸ Checking Ollama (optional)..."
if command -v ollama >/dev/null 2>&1; then
    echo "  ✓ ollama installed"
else
    echo "  → ollama not installed (only needed for local models)"
    echo "    install: https://ollama.com/download"
fi

# --- Verify -------------------------------------------------------------------
echo ""
echo "─────────────────────────────────────────────────────"
if command -v trae >/dev/null 2>&1; then
    echo "✓ trae-agent installed successfully!"
    echo ""
    echo "Quick start:"
    echo "  trae skills list                  # list available skills"
    echo "  trae plan \"add tests for utils\"   # read-only plan first"
    echo "  trae run \"fix the bug in auth\"   # then build & execute"
else
    echo "⚠ Installation finished, but 'trae' is not on PATH."
    echo "  Add this to your shell rc file:"
    echo "    export PATH=\"$(dirname "$INSTALL_BIN"):\$PATH\""
fi
