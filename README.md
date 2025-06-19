
## Run

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

## GPU bench

```sh
python benchmark.py --batch-size 64 --seq-length 100 --learning-rate 0.0005 --loss-function mse --optimizer adam --runs 10 --no-save --gpu cuda
```

or `--gpu mps`


## Docker + Singularity cluster

### Option 1: Build with Docker ➡️ Upload Docker image ➡️ Translate to Singularity

Change `--platform linux/amd64` accordingly.

```sh
docker build --no-cache --platform linux/amd64 -t freeimpala:dev-amd64 -f Dockerfile . && \
docker save freeimpala:dev-amd64 | ssh -p 10022 cluster.uy "cat > ~/img/freeimpala_amd64.tar"
```

then, ssh into the server and:

```sh
singularity build --force $HOME/img/freeimpala_amd64.sif docker-archive://$HOME/img/freeimpala_amd64.tar
```
- then run it: `singularity run freeimpala_amd64.sif`
- or execute a specific command `singularity exec freeimpala_amd64.sif /usr/local/bin/freeimpala`
- or start an interactive shell `singularity shell freeimpala_amd64.sif`

### Option 2: Build with Docker targetting arch ➡️ Extract binary ➡️ Upload bin

```sh
docker build --no-cache --platform linux/amd64 -t freeimpala:dev-amd64 -f Dockerfile .
docker run --rm -v /tmp:/output freeimpala:dev-amd64 cp /app/freeimpala /output/freeimpala
rsync -avz -e "ssh -p 10022" /tmp/freeimpala cluster.uy:~/bin/
```

then, `~/bin/freeimpala` should just work.
