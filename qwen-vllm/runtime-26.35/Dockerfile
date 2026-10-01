# Upstream vLLM v0.30.0 XPU with Intel's GPU compute runtime upgraded from 26.27 to
# 26.35 and IGC, the compiler that turns SYCL kernels into GPU code, from 2.38.2 to
# 2.41.5. Everything else (vLLM, PyTorch, vllm-xpu-kernels, Level Zero loader 1.32,
# gmmlib) is unchanged, so any speed difference comes from the runtime and compiler:
# the SYCL-side counterpart of the Mesa 26.2 test in ../../llamacpp-bench.
# Unlike the llm-scaler image, this base uses Intel's own intel-igc-* packages, so a
# plain dpkg -i upgrades in place.
FROM vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d

ARG NEO=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10
ARG IGC=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5

WORKDIR /tmp/neo
RUN curl -sfLO $IGC/intel-igc-core-2_2.41.5+22716_amd64.deb \
 && curl -sfLO $IGC/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
 && curl -sfLO $NEO/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
 && curl -sfLO $NEO/libze-intel-gpu1_26.35.39758.10-0_amd64.deb \
 && curl -sfLO $NEO/intel-ocloc_26.35.39758.10-0_amd64.deb \
 && printf '%s  %s\n' \
    0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628 intel-igc-core-2_2.41.5+22716_amd64.deb \
    779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058 intel-igc-opencl-2_2.41.5+22716_amd64.deb \
    61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301 intel-opencl-icd_26.35.39758.10-0_amd64.deb \
    c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470 libze-intel-gpu1_26.35.39758.10-0_amd64.deb \
    c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7 intel-ocloc_26.35.39758.10-0_amd64.deb \
    | sha256sum -c - \
 && dpkg -i *.deb \
 && rm -rf /tmp/neo
WORKDIR /
