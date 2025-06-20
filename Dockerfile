FROM ubuntu:24.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libstdc++-12-dev \
        libssl-dev

WORKDIR /app
COPY . /app
RUN rm -rf /app/build

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++ -pthread" && \
    make

FROM ubuntu:24.04

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/freeimpala freeimpala
