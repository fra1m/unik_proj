# FaceID core: build and run in a single (simple) image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Use BuildKit cache mounts so apt downloads and build artifacts are reused.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    nodejs \
    npm \
    ccache \
    libopencv-dev \
    libdlib-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# NOTE: build context must be repo root (..)
COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cmake --build /build --target FaceIDCli -j

WORKDIR /app
COPY faceid-core/server.js ./server.js
RUN mkdir -p /opt/faceid && cp -R /build/bin /opt/faceid/bin

ENV PORT=3010
ENV DATA_PATH=/opt/faceid/bin/data/last_embedding.json
ENV FACEID_CLI_PATH=/opt/faceid/bin/FaceIDCli
ENV FACEID_CLI_CWD=/opt/faceid/bin

EXPOSE 3010
CMD ["node", "server.js"]
