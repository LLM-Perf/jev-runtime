# Template: generate a verified context with python -m deployment.image_context prepare.
# BuildKit's bundled Dockerfile frontend is used; RUN --network=none is required.
FROM docker.io/library/python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49 AS builder
USER 0:0
WORKDIR /opt/jev-build
COPY release.py ./release.py
COPY locked ./locked
RUN --network=none python -I release.py install \
    --bundle /opt/jev-build/locked --manifest-sha256 e377de81cc9a68bdb3eb63afc643cffcfacfe53c70a899ede6e25834580cc0d8 \
    --output /opt/jev/venv
RUN --network=none python -I -c "import os; os.makedirs('/data/jev', exist_ok=True); os.chown('/data/jev', 10001, 10001)"

FROM docker.io/library/python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49
LABEL org.opencontainers.image.revision="6f0cda85f72619193be7a1d1cc89d75142899bf3" \
      io.jev.gateway-lock-sha256="e377de81cc9a68bdb3eb63afc643cffcfacfe53c70a899ede6e25834580cc0d8"
COPY --from=builder /opt/jev/venv /opt/jev/venv
COPY --from=builder --chown=10001:10001 /data/jev /data/jev
COPY container_gateway.py image.json /opt/jev/
ENV PATH="/opt/jev/venv/bin:$PATH" HOME="/data/jev" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JEV_CONFIG="/config/gateway.yaml"
WORKDIR /data/jev
USER 10001:10001
EXPOSE 8795
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=3 \
    CMD ["/opt/jev/venv/bin/python", "-I", "/opt/jev/container_gateway.py", "healthcheck"]
ENTRYPOINT ["/opt/jev/venv/bin/python", "-I", "/opt/jev/container_gateway.py", "serve"]
