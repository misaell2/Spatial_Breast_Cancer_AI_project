FROM mambaorg/micromamba:1.5.10

LABEL project="Spatial_Breast_Cancer_AI_project"
LABEL description="Reproducible Scanpy/Squidpy environment for breast cancer spatial transcriptomics ML workflow"

WORKDIR /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "-c", "import scanpy as sc; import squidpy as sq; import sklearn; print('Python/Scanpy/Squidpy/sklearn environment ready'); print('scanpy', sc.__version__); print('squidpy', sq.__version__); print('sklearn', sklearn.__version__)"]
