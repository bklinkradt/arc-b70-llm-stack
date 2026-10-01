# llama.cpp Vulkan vs SYCL experiment. Run from the repo root with `make -C llamacpp-bench <target>`.
# See README.md for the procedure.

COMPOSE := docker compose --profile vulkan --profile sycl --profile sycl2 --profile vulkan-mesa26
# Path under $MODELS_DIR (default ~/models).
MODEL ?= Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
SPEC ?= none
# Server slots (concurrent requests) and the sycl-2card split mode.
PARALLEL ?= 1
SPLIT ?= tensor
# GGML_SYCL_ENABLE_OPT for the one-card sycl server, and the server's model name.
SYCL_OPT ?= 1
ALIAS ?= Qwen3.8-27B
export MODEL SPEC PARALLEL SPLIT SYCL_OPT ALIAS
BENCH_ARGS := -m /models/$(MODEL) -ngl 999 -fa 1 -p 512,4096 -n 128 -r 5
# none = card 0 alone, the baseline. row crashes at load in this SYCL build (see README).
SPLITS := none layer tensor

.PHONY: help take-card release-card vulkan sycl devices-vulkan devices-sycl \
	llama-bench-vulkan llama-bench-sycl bench-vulkan bench-sycl down \
	vulkan-mesa26 devices-vulkan-mesa26 llama-bench-vulkan-mesa26 bench-vulkan-mesa26 \
	take-both release-both split-bench sycl-2card bench-sycl-2card

help:
	@echo "take-card / release-card      borrow the second B70 from llama-swap, and give it back"
	@echo "vulkan / sycl                 start that backend's server (stops the other)"
	@echo "devices-vulkan / devices-sycl list the GPUs each backend sees"
	@echo "llama-bench-vulkan / -sycl    raw pp512/pp4096/tg128 numbers (server must be down)"
	@echo "bench-vulkan / bench-sycl     ../qwen-vllm/bench.py against that server, args=\"5 code 1 0\""
	@echo "take-both / release-both      stop / restart ALL chat (both cards) for split-bench"
	@echo "split-bench                   SYCL on both cards: -sm none/layer/row/tensor, single stream and batched"
	@echo "sycl-2card                    server on both cards, 127.0.0.1:8092 (SPLIT=tensor, PARALLEL=, SPEC=)"
	@echo "bench-sycl-2card              ../qwen-vllm/bench.py against it"
	@echo "down                          stop and remove the experiment containers"
	@echo "MODEL=<path under $$MODELS_DIR> overrides the GGUF (default $(MODEL))"
	@echo "SPEC=draft-mtp                turns on MTP (3 drafts) for vulkan / sycl"

# Same as `make mtp` in the root: card 0's vLLM keeps serving chat whatever the GPU
# layout; images are unavailable until release-card.
take-card:
	docker stop llama-swap qwen-vllm-2 qwen-vllm-tp2 qwen-image
	docker start qwen-vllm

# llama-swap restores the GPU layout when it starts.
release-card: down
	docker start llama-swap

# Both cards: production chat is down until release-both.
take-both: take-card
	docker stop qwen-vllm

release-both: release-card

# Single-stream (llama-bench) and aggregate throughput at 1-16 parallel sequences
# (llama-batched-bench), the load where vLLM's TP=2 paid off.
split-bench:
	$(COMPOSE) run --rm --no-deps --entrypoint /app/llama-bench llama-sycl-2card \
		-m /models/$(MODEL) -ngl 999 -fa 1 -p 512,4096 -n 128 -r 3 -sm $(shell echo $(SPLITS) | tr ' ' ,)
	for sm in $(SPLITS); do \
		echo "== split-mode $$sm"; \
		$(COMPOSE) run --rm --no-deps --entrypoint /app/llama-batched-bench llama-sycl-2card \
			-m /models/$(MODEL) -ngl 999 -fa on -sm $$sm -c 16384 -npp 512 -ntg 128 -npl 1,4,8,16; \
	done

sycl-2card:
	$(COMPOSE) rm -sf llama-vulkan llama-sycl llama-vulkan-mesa26
	$(COMPOSE) up -d llama-sycl-2card

bench-sycl-2card:
	BENCH_URL=http://localhost:8092 python3 ../qwen-vllm/bench.py $(args)

vulkan:
	$(COMPOSE) rm -sf llama-sycl llama-sycl-2card llama-vulkan-mesa26
	$(COMPOSE) up -d llama-vulkan

sycl:
	$(COMPOSE) rm -sf llama-vulkan llama-sycl-2card llama-vulkan-mesa26
	$(COMPOSE) up -d llama-sycl

# Vulkan on Mesa 26.2 (vulkan-mesa26/Dockerfile), server on 127.0.0.1:8093.
vulkan-mesa26:
	$(COMPOSE) rm -sf llama-vulkan llama-sycl llama-sycl-2card
	$(COMPOSE) up -d llama-vulkan-mesa26

devices-vulkan-mesa26:
	$(COMPOSE) run --rm --no-deps llama-vulkan-mesa26 --list-devices

llama-bench-vulkan-mesa26:
	$(COMPOSE) run --rm --no-deps --entrypoint /app/llama-bench llama-vulkan-mesa26 $(BENCH_ARGS)

bench-vulkan-mesa26:
	BENCH_URL=http://localhost:8093 python3 ../qwen-vllm/bench.py $(args)

devices-vulkan:
	$(COMPOSE) run --rm --no-deps llama-vulkan --list-devices

devices-sycl:
	$(COMPOSE) run --rm --no-deps llama-sycl --list-devices

llama-bench-vulkan:
	$(COMPOSE) run --rm --no-deps --entrypoint /app/llama-bench llama-vulkan $(BENCH_ARGS)

llama-bench-sycl:
	$(COMPOSE) run --rm --no-deps --entrypoint /app/llama-bench llama-sycl $(BENCH_ARGS)

bench-vulkan:
	BENCH_URL=http://localhost:8090 python3 ../qwen-vllm/bench.py $(args)

bench-sycl:
	BENCH_URL=http://localhost:8091 python3 ../qwen-vllm/bench.py $(args)

down:
	$(COMPOSE) down
