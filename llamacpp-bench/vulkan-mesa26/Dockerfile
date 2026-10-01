# The same llama.cpp commit as the ghcr full-vulkan image (b11243, fc07d781e), built on
# Fedora 44 for its newer Mesa ANV (26.2.x vs the Ubuntu image's 26.0.8), so Mesa is
# the only variable. Mesa 26.1 reportedly doubled Vulkan decode on the B70.
FROM fedora:44 AS build
RUN dnf -y install git cmake gcc-c++ ninja-build glslc vulkan-headers vulkan-loader-devel \
      spirv-headers-devel libcurl-devel && dnf clean all
RUN git clone https://github.com/ggml-org/llama.cpp /src && cd /src && git checkout fc07d781e
RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
      -DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /build --target llama-server llama-bench llama-batched-bench -j

FROM fedora:44
RUN dnf -y install mesa-vulkan-drivers vulkan-loader vulkan-tools libcurl libgomp && dnf clean all
COPY --from=build /build/bin/llama-server /build/bin/llama-bench /build/bin/llama-batched-bench /app/
ENTRYPOINT ["/app/llama-server"]
