# Lambda container image for the Seller Support Triage pipeline.
#
# WHY A CONTAINER AND NOT A ZIP
#   1. onnxruntime and numpy ship compiled binaries. Installing them on
#      Windows or macOS fetches wheels for that platform; Lambda runs Amazon
#      Linux on x86-64 and needs manylinux wheels. Building FROM the AWS base
#      image gets the right ones by construction.
#   2. onnxruntime (~45MB) + the bge-small model (~130MB) + numpy + chunks.jsonl
#      is at or over the 250MB unzipped limit for zip deployments. The
#      container limit is 10GB.
#   3. The embedding model has to be baked in — see below.

FROM public.ecr.aws/lambda/python:3.13

# Pinned to the versions the measured numbers were produced against.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# ---------------------------------------------------------------------------
# Bake the embedding model into the image.
#
# search.py instantiates TextEmbedding lazily, and fastembed downloads the
# model from HuggingFace on first use. Without this step every cold start
# would fetch ~130MB over the network before answering its first ticket —
# slow, and a third-party dependency at exactly the wrong moment. Pre-fetching
# at build time means the only cold-start cost is importing fastembed (~2.65s
# measured locally).
#
# FASTEMBED_CACHE_PATH must be set for BOTH the download below and the runtime,
# or the container downloads it again into a path it cannot write.
# ---------------------------------------------------------------------------
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN mkdir -p /opt/fastembed && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" && \
    chmod -R a+rX /opt/fastembed

# ---------------------------------------------------------------------------
# The code. Layout is preserved deliberately: settings.py computes
# ROOT = <its own dir>/.. and derives gate/ and opensearch/ from it, so
# flattening these directories breaks the sys.path bootstrap.
# ---------------------------------------------------------------------------
COPY gate/        ${LAMBDA_TASK_ROOT}/gate/
COPY opensearch/  ${LAMBDA_TASK_ROOT}/opensearch/
COPY pipeline/    ${LAMBDA_TASK_ROOT}/pipeline/
COPY chunks.jsonl ${LAMBDA_TASK_ROOT}/
COPY handler.py   ${LAMBDA_TASK_ROOT}/

# ---------------------------------------------------------------------------
# Runtime configuration.
#
# PIPELINE_DRAFT_TIMEOUT is 15, NOT the 60s default. API Gateway hard-caps an
# integration at 29 seconds. At 60s a hung mantle call would blow that cap and
# return 504 BEFORE the drafter's fallback chain could degrade to the template
# — turning a handled failure into a dead demo. At 15s the fallback fires with
# room to spare. Drafts measure ~2.4s, so this never triggers in the happy path.
# ---------------------------------------------------------------------------
ENV PIPELINE_RETRIEVER=local \
    PIPELINE_CLASSIFIER=local \
    PIPELINE_DRAFTER=mantle \
    PIPELINE_TRAIL=dynamodb \
    PIPELINE_TABLE=seller-triage-tickets \
    PIPELINE_DRAFT_TIMEOUT=15

CMD ["handler.lambda_handler"]
