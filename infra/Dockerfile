# The training image. One dependency source of truth, two delivery mechanisms.
#
# This installs from requirements/train-linux-x86_64-cu121.txt — the same lock the local
# virtual environment uses. An image that drifts from that lock is a defect, and CI
# asserts the two agree, because a container whose contents nobody can name reproduces
# nothing.
#
# Built for linux/amd64 regardless of the machine that builds it: the cluster is x86-64,
# and an image built natively on Apple silicon would be arm64 and fail to start there
# without any obvious explanation. The platform is passed to `docker build` by
# scripts/build_image.sh rather than pinned here, which is what buildkit expects and
# leaves the file usable on either architecture.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Determinism settings, baked in rather than left to whoever submits the job.
# Multithreaded BLAS combines partial sums in completion order, so a reduction is correct
# but not bit-identical between runs; pinning the thread counts removes that source. The
# hash seed must be set before the interpreter starts, which a Dockerfile can do and a
# Python entry point cannot.
# The allocator hint the OOM message itself recommends. A run that fits at a given
# micro-batch can still die on fragmentation, and the heaviest architecture reported 1.83
# GiB reserved-but-unallocated at the moment it failed.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Python 3.12 exactly, which this base image does not carry: the CUDA 12.1 images are
# built on Ubuntu 22.04, whose archive tops out at Python 3.10. The deadsnakes PPA is the
# ordinary way to get 3.12 there, and it is preferred over moving to a newer Ubuntu base
# because that would mean a different CUDA version and a regenerated lock — a larger change
# than the problem warrants.
#
# libgomp is the OpenMP runtime XGBoost and LightGBM link against. It ships with gcc on
# most distributions but not in this base image, and its absence produces an import failure
# that looks nothing like a missing system library — the same trap that cost time on macOS
# with libomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# `python3-pip` from the archive would install pip for 3.10, not 3.12, and the wheels it
# then resolved would be for the wrong interpreter. ensurepip bootstraps against the
# interpreter actually in use.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && python -m ensurepip --upgrade

# The lock is resolved for 3.12 exactly. A mismatch here would install wheels built for a
# different interpreter and fail much later, in a place that would not point back to this.
RUN python -c "import sys; assert sys.version_info[:2] == (3, 12), \
    f'image has Python {sys.version_info[:2]}, the lock targets 3.12'"

WORKDIR /workspace

# Dependencies before source, so editing the package does not reinstall three gigabytes of
# CUDA wheels on every build.
COPY requirements/train-linux-x86_64-cu121.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
RUN python -m pip install -e . --no-deps

# The GPT-2 tokenizer, baked in. Jobs run with HF_HUB_OFFLINE=1 so a training run cannot
# quietly fetch anything, which means every asset it needs has to already be here — and the
# tokenizer defines the vocabulary the whole benchmark is measured against, so it is exactly
# the thing that must not vary between a run today and a re-run next year.
#
# Downloaded at build time and then asserted to load offline. Downloading alone would not
# prove it: the download populates a cache whose location depends on HF_HOME, and a cache
# written to one path and read from another looks identical until the network is gone.
ENV HF_HOME=/opt/hf
RUN python -c "\
from transformers import GPT2TokenizerFast; \
GPT2TokenizerFast.from_pretrained('gpt2')" \
    && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "\
from transformers import GPT2TokenizerFast; \
t = GPT2TokenizerFast.from_pretrained('gpt2'); \
assert t.vocab_size == 50257, f'vocabulary is {t.vocab_size}, the cost model assumes 50257'; \
print('tokenizer loads offline, vocabulary', t.vocab_size)"

# The lock's hash travels with the image, so a manifest can record which dependency set
# produced a result rather than only which image tag did — tags move, hashes do not.
COPY requirements/train-linux-x86_64-cu121.txt /workspace/requirements.lock
RUN python -c "import hashlib,pathlib; \
    print(hashlib.sha256(pathlib.Path('/workspace/requirements.lock').read_bytes()).hexdigest())" \
    > /workspace/requirements.lock.sha256

# The image and the lock are two descriptions of one environment, and building the image
# does not enforce that they agree. A resolver that picked a different version, a layer
# restored from a stale cache, a package fixed by hand and never written back — each leaves
# an image that runs and produces numbers while the manifest records a lock hash that
# truthfully describes a file no longer describing this container. A broken image fails; a
# drifted one publishes.
RUN python -m nas4textgen.audit_image \
    || (echo "the image does not match the lock it was built from" && exit 1)

# Fail at build time if the environment cannot actually run, rather than on a cluster node
# twenty minutes into a job. A successful pip install does not prove a native library
# loads.
RUN python -m nas4textgen.check_env || \
    (echo "the image cannot run its own environment check" && exit 1)

# Verify the entrypoint resolves at build time. A build does not run its own ENTRYPOINT,
# so a module named here but never written fails on the first line of a job — after a node
# has been allocated and three gigabytes pulled. This caught exactly that.
RUN python -m nas4textgen.train --help > /dev/null \
    || (echo "the ENTRYPOINT module does not resolve" && exit 1)

# And that it still accepts the arguments the submitter sends. Azure ML chooses where it
# mounts the corpus and the output directory, so both arrive as arguments; an image whose
# entry point stopped accepting them would fail once per job, on every node, after the pull.
RUN python -m nas4textgen.train --help | grep -q -- "--cache" \
    && python -m nas4textgen.train --help | grep -q -- "--output" \
    || (echo "the entry point no longer accepts the mount arguments jobs are given" && exit 1)

ENTRYPOINT ["python", "-m", "nas4textgen.train"]
