# llama-swap's own images bundle llama.cpp; here it only needs its binary plus the
# Docker CLI to start and stop the model containers.
FROM docker:29.8.1-cli@sha256:018edbc908e08fcc9dbf029c812c34251e9b4719e6f71ca0e5eae2a987d014ca

ARG LLAMA_SWAP_VERSION=260
ARG LLAMA_SWAP_SHA256=d856a908507560cbdc253300bcf49092c7ead3c85098687428b0c9d4832ff46d
RUN wget -qO /tmp/llama-swap.tgz \
      "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz" \
 && echo "${LLAMA_SWAP_SHA256}  /tmp/llama-swap.tgz" | sha256sum -c - \
 && tar -xzf /tmp/llama-swap.tgz -C /usr/local/bin llama-swap \
 && rm /tmp/llama-swap.tgz

COPY gpu-layout.sh /usr/local/bin/gpu-layout
COPY run-container.sh /usr/local/bin/run-container
COPY stop-container.sh /usr/local/bin/stop-container
COPY entrypoint.sh /usr/local/bin/entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint"]
