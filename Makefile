# AI Skill Planner - Development & Deployment Makefile

# Variables
IMAGE_NAME = ai-skill-planner
CONTAINER_NAME = ai-skill-planner-api
PORT = 8000
CONDA_ENV = ai_skill_planner

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help setup test clean docker-build docker-run docker-stop dev

help: ## Show this help message
	@echo "$(GREEN)AI Skill Planner - Available Commands$(NC)"
	@echo "================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Setup conda environment and initialize database
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@if command -v conda >/dev/null 2>&1; then \
		chmod +x scripts/setup_environment.sh && ./scripts/setup_environment.sh; \
	else \
		echo "$(RED)Conda not found. Installing dependencies with pip...$(NC)"; \
		pip install -r requirements.txt; \
		python database/init_db.py; \
		python data/skills_taxonomy.py; \
		python data/generate_people.py; \
		python data/generate_projects.py; \
		python data/generate_assignments.py; \
		python data/generate_evidence.py; \
	fi
	@echo "$(GREEN)Setup complete!$(NC)"

test: ## Run API tests with pytest
	@echo "$(GREEN)Running API tests...$(NC)"
	@pytest --maxfail=1 --disable-warnings

dev: ## Start development server
	@echo "$(GREEN)Starting development server...$(NC)"
	@if command -v conda >/dev/null 2>&1 && conda env list | grep -q $(CONDA_ENV); then \
		conda run -n $(CONDA_ENV) uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT); \
	else \
		uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT); \
	fi

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t $(IMAGE_NAME) .
	@echo "$(GREEN)Docker image built: $(IMAGE_NAME)$(NC)"

docker-run: ## Run Docker container
	@echo "$(GREEN)Starting Docker container...$(NC)"
	@docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--restart unless-stopped \
		$(IMAGE_NAME)
	@echo "$(GREEN)Container started: http://localhost:$(PORT)$(NC)"

docker-stop: ## Stop and remove Docker container
	@echo "$(YELLOW)Stopping Docker container...$(NC)"
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "$(GREEN)Container stopped and removed$(NC)"

docker-logs: ## View Docker container logs
	@docker logs -f $(CONTAINER_NAME)

docker-compose-up: ## Start with docker-compose
	@echo "$(GREEN)Starting with docker-compose...$(NC)"
	@docker-compose up --build -d
	@echo "$(GREEN)Services started: http://localhost:$(PORT)$(NC)"

docker-compose-down: ## Stop docker-compose services
	@echo "$(YELLOW)Stopping docker-compose services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)Services stopped$(NC)"

clean: ## Clean up temporary files and containers
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	@rm -rf .pytest_cache/
	@rm -rf *.log logs/
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker system prune -f
	@echo "$(GREEN)Cleanup complete$(NC)"

install: ## Install in current Python environment
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@pip install -r requirements.txt

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@python -m black . --line-length 88 --exclude "venv|env|build|dist"

lint: ## Lint code with flake8
	@echo "$(GREEN)Linting code...$(NC)"
	@python -m flake8 . --select=E9,F63,F7,F82 --exclude=venv,env,build,dist

health-check: ## Check if API is running
	@echo "$(GREEN)Checking API health...$(NC)"
	@curl -f http://localhost:$(PORT)/health || echo "$(RED)API not responding$(NC)"

db-reset: ## Reset database with fresh sample data
	@echo "$(YELLOW)Resetting database...$(NC)"
	@python database/init_db.py
	@python data/skills_taxonomy.py
	@python data/generate_people.py
	@python data/generate_projects.py
	@python data/generate_assignments.py
	@python data/generate_evidence.py
	@echo "$(GREEN)Database reset complete$(NC)"

# Quick development workflow
quick-start: setup test dev ## Setup, test, and start dev server

# Production deployment
deploy: docker-build docker-compose-up ## Build and deploy with docker-compose

# Complete cleanup and fresh start
fresh-start: clean setup test ## Clean everything and start fresh
