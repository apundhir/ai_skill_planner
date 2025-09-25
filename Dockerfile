# AI Skill Planner - Production Docker Image
FROM continuumio/miniconda3:23.10.0-1

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONDA_ENV_NAME=ai_skill_planner

# Copy environment configuration
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ai_skill_planner", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Initialize database with sample data
RUN conda run -n ai_skill_planner python database/init_db.py && \
    conda run -n ai_skill_planner python data/skills_taxonomy.py && \
    conda run -n ai_skill_planner python data/generate_people.py && \
    conda run -n ai_skill_planner python data/generate_projects.py && \
    conda run -n ai_skill_planner python data/generate_assignments.py && \
    conda run -n ai_skill_planner python data/generate_evidence.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD conda run -n ai_skill_planner python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "ai_skill_planner", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]