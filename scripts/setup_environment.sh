#!/bin/bash
# Setup script for AI Skill Planner development environment

set -e  # Exit on any error

echo "🚀 Setting up AI Skill Planner development environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
if conda env list | grep -q "ai_skill_planner"; then
    echo "⚠️  Environment 'ai_skill_planner' already exists. Removing it..."
    conda env remove -n ai_skill_planner -y
fi

conda env create -f environment.yml

echo "✅ Conda environment created successfully!"

# Activate environment and initialize database
echo "🗄️  Initializing database with sample data..."
conda run -n ai_skill_planner python database/init_db.py
conda run -n ai_skill_planner python data/skills_taxonomy.py
conda run -n ai_skill_planner python data/generate_people.py
conda run -n ai_skill_planner python data/generate_projects.py
conda run -n ai_skill_planner python data/generate_assignments.py
conda run -n ai_skill_planner python data/generate_evidence.py

echo "✅ Database initialized with sample data!"

# Run tests if available
if [ -d "tests" ]; then
    echo "🧪 Running tests..."
    conda run -n ai_skill_planner python -m pytest tests/ -v
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate ai_skill_planner"
echo ""
echo "To start the development server:"
echo "  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To build and run with Docker:"
echo "  docker-compose up --build"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"