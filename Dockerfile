# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest

# — build args —
ARG MPI=openmpi:5.0.5
ARG LIBTORCH_VERSION=2.7.1
ARG LIBTORCH_DIR=/opt/libtorch

# — install MPI, cmake, compilers —
RUN MPI_IMPL=$(echo $MPI | cut -d: -f1) && \
    MPI_VERSION=$(echo $MPI | cut -d: -f2) && \
    if [ -z "$MPI_IMPL" ] || [ -z "$MPI_VERSION" ]; then \
      echo "MPI arg must be impl:version" && exit 1; \
    fi && \
    if [ "$MPI_IMPL" = "openmpi" ]; then \
      conda install -c conda-forge openmpi=${MPI_VERSION} cmake make compilers; \
    elif [ "$MPI_IMPL" = "mpich" ]; then \
      conda install -c conda-forge mpich=${MPI_VERSION} cmake make compilers; \
    else \
      echo "Unsupported MPI: $MPI_IMPL" && exit 1; \
    fi && \
    conda clean -afy

# — system deps —
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libssl-dev wget unzip libpaho-mqtt-dev && \
    rm -rf /var/lib/apt/lists/*

# — fetch libtorch for the right arch —
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"; \
    else \
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"; \
    fi && \
    echo "Downloading LibTorch from $URL ..." && \
    wget -qO /tmp/libtorch.zip "$URL" && \
    mkdir -p ${LIBTORCH_DIR} && \
    unzip -q /tmp/libtorch.zip -d /tmp && \
    mv /tmp/libtorch/* ${LIBTORCH_DIR} && \
    rm -rf /tmp/libtorch* /tmp/libtorch.zip

# — copy your code in —
WORKDIR /app
COPY . /app

ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

# — generate MPI‐env script —
RUN MPI_IMPL=$(echo $MPI | cut -d: -f1) && \
    if [ "$MPI_IMPL" = "openmpi" ]; then \
      printf "export OMPI_ALLOW_RUN_AS_ROOT=1\nexport OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1\n" > /set_mpi_env.sh; \
    else \
      touch /set_mpi_env.sh; \
    fi

# — build everything with MPI compilers and libtorch —
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    mkdir -p build && cd build && \
    cmake \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_PREFIX_PATH=${LIBTORCH_DIR} \
      .. && \
    cmake --build . --parallel

# — install binaries & cleanup —
RUN mkdir -p /usr/local/bin && \
    mv /app/build/* /usr/local/bin/ && \
    rm -rf /app /opt/conda/pkgs

# — sanity checks —
RUN mpirun --version
RUN ls -l /usr/local/bin || true

# — entrypoint —
CMD ["/bin/bash", "-c", ". /set_mpi_env.sh && exec /bin/bash"]
