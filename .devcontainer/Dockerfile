# ---- BASE IMAGE ----
# Microsoft official Python Devcontainer for full Windows compatibility
FROM mcr.microsoft.com/devcontainers/python:3.12

# ---- PYTHON SETTINGS ----
ENV PYTHONUNBUFFERED=1

# ---- INSTALL OS DEPENDENCIES ----
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- PIP UPGRADE ----
RUN pip install --upgrade pip

# ---- INSTALL POETRY ----
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (IMPORTANT FOR WINDOWS)
ENV PATH="/root/.local/bin:${PATH}"

# ---- SET WORKING DIRECTORY ----
WORKDIR /workspace
