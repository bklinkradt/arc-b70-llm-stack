# Day-to-day commands for the whole stack. See README.md.

COMPOSE := docker compose
# The model containers. llama-swap starts and stops them to fit the GPU layout
# (gateway/llama-swap/layouts/layouts.conf), so they are only ever created here.
MANAGED := vllm vllm-2 vllm-tp2 qwen-image
LAYOUTS := gateway/llama-swap/layouts

.PHONY: help network build up down restart ps logs bench ddns mode mtp mtp-down

help:
	@echo "make build          build local images (vLLM runtime, qwen-image, llama-swap)"
	@echo "make up             start the stack (creates llm-net and the managed containers)"
	@echo "make down           stop and remove containers; volumes are kept"
	@echo "make restart s=X    restart one service"
	@echo "make ps             container status"
	@echo "make logs [s=X]     follow logs, optionally for one service"
	@echo "make bench          decode benchmark against qwen-vllm (see qwen-vllm/README.md)"
	@echo "make ddns           start the optional dynamic DNS updater"
	@echo "make mode [M=X]     show the GPU layout, or switch to X (split, tp2)"
	@echo "make mtp MTP_N=4    MTP draft-depth experiment on the second card (:8002); mtp-down reverts"

network:
	@docker network inspect llm-net >/dev/null 2>&1 || docker network create llm-net

build:
	$(COMPOSE) --profile managed build

# `create` recreates a model container whose config changed, which leaves it stopped;
# the reconcile starts whatever the layout wants. On a first start the entrypoint
# reconciles instead, and the exec can fail while llama-swap is still starting.
up: network
	$(COMPOSE) --profile managed create $(MANAGED)
	$(COMPOSE) up -d
	docker exec llama-swap gpu-layout reconcile || echo "llama-swap will apply the layout when it starts"

# Never add -v here: the volumes hold LiteLLM's keys and the TLS certificates.
down:
	$(COMPOSE) --profile managed --profile ddns down

restart:
	$(COMPOSE) restart $(s)

ps:
	$(COMPOSE) --profile managed --profile ddns ps -a

logs:
	$(COMPOSE) --profile managed --profile ddns logs -f --tail=100 $(s)

bench:
	cd qwen-vllm && python3 bench.py $(args)

ddns: network
	$(COMPOSE) --profile ddns up -d ddns

# Saves the layout and applies it now. An on-demand model that is loaded (qwen-image)
# keeps its card; the new layout fills the rest and takes over once it unloads.
mode:
ifdef M
	@grep -qE '^layout +$(M)( |$$)' $(LAYOUTS)/layouts.conf || \
		{ echo "unknown layout '$(M)'; see $(LAYOUTS)/layouts.conf"; exit 1; }
	echo $(M) > $(LAYOUTS)/mode
	docker exec llama-swap gpu-layout reconcile
endif
	@docker exec llama-swap gpu-layout status
	@echo "layouts: $$(awk '$$1 == "layout" { printf "%s ", $$2 }' $(LAYOUTS)/layouts.conf)"

# Draft depth for `make mtp`; production (qwen-vllm/docker-compose.yml) uses 3.
MTP_N ?= 3
export MTP_N
MTP_COMPOSE := $(COMPOSE) -f compose.yaml -f qwen-vllm/docker-compose.mtp.yml --profile mtp

# Borrows the second card: llama-swap stops, card 0 keeps serving chat on qwen-vllm
# whatever the layout, and images are unavailable until mtp-down.
mtp:
	docker stop llama-swap qwen-vllm-2 qwen-vllm-tp2 qwen-image
	docker start qwen-vllm
	$(MTP_COMPOSE) up -d --force-recreate vllm-mtp

mtp-down:
	$(MTP_COMPOSE) rm -sf vllm-mtp
	docker start llama-swap

# Maintainer-only targets (make publish); absent from the public repo.
-include publish.mk
