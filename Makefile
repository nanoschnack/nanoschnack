.PHONY: help setup install bootstrap sync activate precommit dev clean deps check all

# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv
PRE_COMMIT = pre-commit
BOOTSTRAP_URL = https://raw.githubusercontent.com/nanoschnack/nanoschnack/refs/heads/main/scripts/bootstrap.sh

# Colors
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Help target (default)
help:
	@echo "$(GREEN)Development Environment Setup$(NC)"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  $(YELLOW)make setup$(NC)        - Complete setup (install uv → bootstrap → dependencies)"
	@echo "  $(YELLOW)make all$(NC)          - Alias for 'setup'"
	@echo "  $(YELLOW)make install-uv$(NC)   - Install uv package manager"
	@echo "  $(YELLOW)make bootstrap$(NC)    - Run bootstrap script"
	@echo "  $(YELLOW)make sync$(NC)         - Install dependencies"
	@echo "  $(YELLOW)make activate$(NC)     - Activate virtual environment"
	@echo "  $(YELLOW)make precommit$(NC)    - Install pre-commit hooks"
	@echo "  $(YELLOW)make dev$(NC)          - Setup for development (sync + precommit)"
	@echo "  $(YELLOW)make check$(NC)        - Check system dependencies"
	@echo "  $(YELLOW)make deps$(NC)         - Check and install dependencies"
	@echo "  $(YELLOW)make clean$(NC)        - Clean up virtual environment"
	@echo "  $(YELLOW)make help$(NC)         - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  $$ $(YELLOW)make setup$(NC)"
	@echo ""
	@echo "Manual steps after setup:"
	@echo "  $$ $(YELLOW)source .venv/bin/activate$(NC)"

# Check system dependencies
check:
	@echo "$(GREEN)Checking system dependencies...$(NC)"
	@command -v curl >/dev/null 2>&1 || { echo "$(RED)Error: curl is not installed$(NC)"; exit 1; }
	@echo "$(GREEN)✓ curl is installed$(NC)"
	@echo "$(GREEN)All dependencies satisfied$(NC)"

# Install uv
install-uv: check
	@echo "$(GREEN)Installing uv...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(GREEN)✓ uv is already installed$(NC)"; \
	else \
		echo "Installing uv from astral.sh..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		echo "$(GREEN)✓ uv installed successfully$(NC)"; \
	fi

# Run bootstrap script
bootstrap: install-uv
	@echo "$(GREEN)Running bootstrap script...$(NC)"
	@curl -fsSL $(BOOTSTRAP_URL) | bash
	@echo "$(GREEN)✓ Bootstrap completed$(NC)"

# Install dependencies
sync: bootstrap
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(UV) sync --extra dev
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Activate virtual environment (note: this runs in a sub-shell)
activate: sync
	@echo "$(GREEN)Virtual environment is ready!$(NC)"
	@echo ""
	@echo "To activate, run:"
	@echo "  $$ $(YELLOW)source .venv/bin/activate$(NC)"
	@echo ""
	@echo "Or use:"
	@echo "  $$ $(YELLOW)make shell$(NC)"

# Install pre-commit hooks
precommit: sync
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	@if [ -f "$(VENV)/bin/pre-commit" ]; then \
		$(VENV)/bin/pre-commit install; \
		echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"; \
	else \
		echo "$(YELLOW)Warning: pre-commit not found in virtual environment$(NC)"; \
	fi

# Development setup (dependencies + pre-commit)
dev: precommit
	@echo "$(GREEN)Development setup complete!$(NC)"

# Complete setup (all steps)
setup: dev activate
	@echo ""
	@echo "$(GREEN)=========================================$(NC)"
	@echo "$(GREEN)🎉 Setup completed successfully!$(NC)"
	@echo "$(GREEN)=========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Activate virtual environment:"
	@echo "     $$ $(YELLOW)source .venv/bin/activate$(NC)"
	@echo "  2. Run pre-commit on all files:"
	@echo "     $$ $(YELLOW)pre-commit run --all-files$(NC)"
	@echo "  3. Start developing!"
	@echo ""
	@echo "To deactivate the environment later:"
	@echo "  $$ $(YELLOW)deactivate$(NC)"

# Alias for setup
all: setup

# Start a shell with activated environment
shell: sync
	@echo "$(GREEN)Starting shell with activated virtual environment...$(NC)"
	@echo "$(YELLOW)Type 'exit' to leave this shell$(NC)"
	@bash --init-file <(echo "source $(VENV)/bin/activate")

# Check dependencies
deps: check
	@echo "$(GREEN)Checking Python dependencies...$(NC)"
	@if [ -f "$(VENV)/bin/python" ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
		$(VENV)/bin/python --version; \
	else \
		echo "$(YELLOW)Virtual environment not found$(NC)"; \
	fi

# Clean up
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		rm -rf $(VENV); \
		echo "$(GREEN)✓ Virtual environment removed$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment not found$(NC)"; \
	fi
	@if [ -f ".pre-commit-config.yaml" ]; then \
		if command -v pre-commit >/dev/null 2>&1; then \
			pre-commit uninstall || true; \
		fi; \
	fi
	@echo "$(GREEN)✓ Cleanup complete$(NC)"
