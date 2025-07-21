ARG nvidia_cuda_version=12.8.1-devel-ubuntu24.04
ARG BASE_IMAGE_TYPE=gpu

# Choose base image based on GPU availability
FROM nvidia/cuda:${nvidia_cuda_version} AS gpu-base
FROM ubuntu:24.04 AS cpu-base

FROM ${BASE_IMAGE_TYPE}-base AS final

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=Asia/Tokyo

SHELL ["/bin/bash", "-c"]

# Install basic dependencies
RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update -y && \
  apt-get install -y --no-install-recommends \
  wget \
  curl \
  sudo \
  software-properties-common \
  lsb-release \
  git \
  ca-certificates \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  kmod \
  libnuma-dev \
  libopenblas-dev \
  libomp-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

WORKDIR /app

# Create a non-root user that matches the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user
ARG BASE_IMAGE_TYPE

# Create group first, handle existing group
RUN getent group ${GROUP_ID} >/dev/null 2>&1 || groupadd -g ${GROUP_ID} ${USERNAME}

# Create user, handle existing user
RUN if ! getent passwd ${USER_ID} >/dev/null 2>&1; then \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}; \
    else \
        existing_user=$(getent passwd ${USER_ID} | cut -d: -f1); \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $existing_user 2>/dev/null || true; \
    fi

# Add user to sudoers
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up directories with proper permissions (minimal chown)
RUN mkdir -p /home/${USERNAME}/.local/share && \
    chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME} && \
    chown ${USER_ID}:${GROUP_ID} /app

# Switch to non-root user
USER ${USERNAME}

# Setup user-specific configurations
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

# Install Python 3.12 and set it as the default
ENV UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv
RUN uv python install 3.12 && \
    uv python pin 3.12 && \
    uv venv

# Setup uv for the user - use container-specific venv location
COPY --chown=${USER_ID}:${GROUP_ID} ./pyproject.toml uv.lock .python-version /app/
RUN uv sync

RUN echo "source /home/${USERNAME}/.venv/bin/activate" >> ~/.bashrc

CMD ["/bin/bash"]
