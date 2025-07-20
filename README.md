
## Run

### Threaded version

```sh
./build/freeimpala \
    --players 1 \
    --iterations 32 \
    --buffer-capacity 32 \
    --batch-size 32 \
    --learner-time 1000 \
    --agents 4 \
    --agent-time 1000
```

### MPI version

```sh
mpirun -n 5 \
    ./build/freeimpala_mpi \
    --players 2 \
    --iterations 320 \
    --buffer-capacity 32 \
    --batch-size 32 \
    --learner-time 100 \
    --agent-time 100 \
    --seed 42
```

Notice that there are no `--agents` flags in the MPI version. In this version 
the number of agents is automatically derived as `n - 1`.


## Compile

### Option 1: Build with Docker ➡️ Upload Docker image ➡️ Translate to Singularity

Change `--platform linux/amd64` accordingly.

```sh
docker build --no-cache --platform linux/amd64 -t freeimpala:dev-amd64 -f Dockerfile . && \
docker save freeimpala:dev-amd64 | ssh -p 10022 cluster "cat > ~/img/freeimpala_amd64.tar"
```

then, ssh into the server and:

```sh
singularity build --force $HOME/img/freeimpala_amd64.sif docker-archive://$HOME/img/freeimpala_amd64.tar
```
- then run it: `singularity run freeimpala_amd64.sif`
- or execute a specific command `singularity exec freeimpala_amd64.sif /usr/local/bin/freeimpala`
- or start an interactive shell `singularity shell freeimpala_amd64.sif`

### Option 2: Build with Docker targetting arch ➡️ Extract binary ➡️ Upload bin

Get cross-compilation by matching target platform with `--platform linux/amd64`.

```sh
docker build --no-cache --platform linux/amd64 -t freeimpala:dev-amd64 -f Dockerfile .
docker run --rm -v /tmp:/output freeimpala:dev-amd64 cp /app/freeimpala /output/freeimpala
rsync -avz -e "ssh -p 10022" /tmp/freeimpala cluster:~/bin/
```

For MPI use

```sh
docker build --no-cache --progress=plain --platform linux/amd64 -t freeimpala:dev-amd64-openmpi -f Dockerfile.OpenMPI --build-arg OPENMPI_VERSION=5.0.5 .
docker run --rm -v /tmp:/output freeimpala:dev-amd64-openmpi cp /usr/local/bin/freeimpala_mpi /output/freeimpala_mpi
rsync -avz -e "ssh -p 10022" /tmp/freeimpala_mpi cluster:~/bin/
```
then, `~/bin/{freeimpala,freeimpala_mpi}` should just work.

Warning: When using OpenMPI don't forget to add `module load mpi/openmpi-x86_64` to your `~/.bashrc` (or equivalent).

### Option 3: Compile manually

Single-machine freeimpala:

```sh
g++ -g -O3 -DNDEBUG -I./include -I./vendor -std=c++17 -Wall -Wextra ./cmd/freeimpala/main.cpp -o freeimpala -lstdc++fs -pthread
```

Multi-machine MPI-based freimpala (e.g., `freeimpala_mpi_sync`)

```sh
mpicxx -g -O3 -DNDEBUG -DUSE_MPI -I./include -I./vendor -std=c++17 -Wall -Wextra ./cmd/freeimpala_mpi_sync/main_mpi_sync.cpp -o freeimpala_mpi_sync -lstdc++fs -pthread
```

## GPU bench

```sh
python benchmark.py \
    --batch-size 64 \
    --seq-length 100 \
    --learning-rate 0.0005 \
    --loss-function mse \
    --optimizer adam \
    --runs 10 \
    --no-save \
    --gpu cuda
```

for `--gpu` use `mps`, `cpu`, `cuda` or `auto`