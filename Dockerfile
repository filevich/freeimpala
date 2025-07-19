# Use continuumio/miniconda3 as the base image
FROM continuumio/miniconda3:latest

# Define build argument for MPI implementation and version (default: openmpi:5.0.5)
# Alternatively, `MPI=mpich`
ARG MPI=openmpi:5.0.5

# Parse the MPI argument to extract implementation and version
RUN MPI_IMPL=$(echo $MPI | cut -d: -f1) && \
    MPI_VERSION=$(echo $MPI | cut -d: -f2) && \
    if [ -z "$MPI_IMPL" ] || [ -z "$MPI_VERSION" ]; then \
        echo "Error: MPI argument must be in the form 'impl:version'" && exit 1; \
    fi && \
    if [ "$MPI_IMPL" != "openmpi" ] && [ "$MPI_IMPL" != "mpich" ]; then \
        echo "Error: Invalid MPI implementation. Use 'openmpi' or 'mpich'." && exit 1; \
    fi && \
    # Install the specified MPI implementation, CMake, make, and compilers via conda
    if [ "$MPI_IMPL" = "openmpi" ]; then \
        conda install -c conda-forge \
            openmpi=${MPI_VERSION} \
            cmake \
            make \
            compilers ; \
    elif [ "$MPI_IMPL" = "mpich" ]; then \
        conda install -c conda-forge \
            mpich=${MPI_VERSION} \
            cmake \
            make \
            compilers ; \
    fi && \
    conda clean -afy

# Install additional system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Set working directory and copy application code
WORKDIR /app
COPY . /app

# Ensure Conda's bin and lib directories are in PATH and LD_LIBRARY_PATH
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

# Generate a script to set MPI-specific environment variables
RUN MPI_IMPL=$(echo $MPI | cut -d: -f1) && \
    if [ "$MPI_IMPL" = "openmpi" ]; then \
        echo "export OMPI_ALLOW_RUN_AS_ROOT=1" > /set_mpi_env.sh && \
        echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> /set_mpi_env.sh; \
    else \
        touch /set_mpi_env.sh; \
    fi

# Activate Conda environment and compile the application
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_CXX_COMPILER=mpicxx \
          -DCMAKE_C_COMPILER=mpicc \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_VERBOSE_MAKEFILE=ON .. && \
    cmake --build .

# Move compiled binaries to /usr/local/bin and clean up
RUN mkdir -p /usr/local/bin/ && \
    mv /app/build/* /usr/local/bin/ && \
    rm -rf /app

# Verify MPI version (build-time check)
RUN mpirun --version

# Verify binaries exist (build-time check)
RUN ls -l /usr/local/bin/ || true

# Set entrypoint to source the environment script and start an interactive shell
CMD ["/bin/bash", "-c", ". /set_mpi_env.sh && /bin/bash"]
