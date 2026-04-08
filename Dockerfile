FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    "pydantic>=2.0" \
    "openai>=1.0" \
    openenv-core

# Copy source
COPY models.py          ./
COPY env.py             ./
COPY baseline_agents.py ./
COPY inference.py       ./
COPY tasks/             ./tasks/
COPY openenv.yaml       ./
COPY README.md          ./

# Runtime environment variables (override with -e flags at docker run)
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV HF_TOKEN=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
ENV AGRI_TASK=hard
ENV AGRI_SCENARIO=default

EXPOSE 8000

# Validate environment loads on build
RUN python -c "from env import AgriEnv; e=AgriEnv(); e.reset(); print('ENV OK')"

CMD ["python", "inference.py"]
