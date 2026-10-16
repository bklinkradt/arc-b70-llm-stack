#!/usr/bin/env python3
"""Generate monitoring/grafana/dashboards/vllm.json (Grafana reloads it within ~10 s)."""
import json
import os
DS={"type":"prometheus","uid":"prometheus"}
# Reference categorical order (dark-mode steps); fixed order, never cycled.
C=["#3987e5","#d95926","#199e70","#c98500"]
GOOD,WARN,CRIT="#0ca30c","#fab219","#d03b3b"
RI="$__rate_interval"
panels=[];pid=[0]
def nid(): pid[0]+=1; return pid[0]
def row(title,y): panels.append({"type":"row","title":title,"id":nid(),"collapsed":False,"gridPos":{"h":1,"w":24,"x":0,"y":y},"panels":[]})
def stat(title,expr,unit,x,y,w=4,decimals=1,desc="",thresholds=None,color=C[0]):
    steps=thresholds or [{"color":color,"value":None}]
    panels.append({"type":"stat","title":title,"description":desc,"id":nid(),"datasource":DS,
      "gridPos":{"h":4,"w":w,"x":x,"y":y},
      "targets":[{"refId":"A","expr":expr,"instant":False,"range":True,"datasource":DS}],
      "fieldConfig":{"defaults":{"unit":unit,**({"decimals":decimals} if decimals is not None else {}),"color":{"mode":"thresholds"},
        "thresholds":{"mode":"absolute","steps":steps},"noValue":"idle"},"overrides":[]},
      "options":{"reduceOptions":{"calcs":["lastNotNull"],"fields":"","values":False},
        "colorMode":"value","graphMode":"area","textMode":"value","justifyMode":"center","orientation":"auto"}})
def ts(title,targets,unit,x,y,w=12,h=8,desc="",minv=0,maxv=None,decimals=None,series=None,stack=False):
    # series: names of label-driven series (legend "card {{device}}") to colour in order.
    names=series or [t[1] for t in targets]
    ov=[{"matcher":{"id":"byName","options":n},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":C[i]}}]} for i,n in enumerate(names)]
    d={"unit":unit,"min":minv,"color":{"mode":"fixed","fixedColor":C[0]},
       "custom":{"lineWidth":2,"fillOpacity":8,"showPoints":"never","spanNulls":30000,"axisSoftMin":0,"gradientMode":"none"}}
    if stack:  # stacked areas: the top edge is the combined total
        d["custom"].update({"stacking":{"mode":"normal","group":"A"},"fillOpacity":35})
    if maxv is not None: d["max"]=maxv
    if decimals is not None: d["decimals"]=decimals
    panels.append({"type":"timeseries","title":title,"description":desc,"id":nid(),"datasource":DS,
      "gridPos":{"h":h,"w":w,"x":x,"y":y},
      "targets":[{"refId":chr(65+i),"expr":e,"legendFormat":l,"datasource":DS} for i,(e,l) in enumerate(targets)],
      "fieldConfig":{"defaults":d,"overrides":ov if len(names)>1 else []},
      "options":{"legend":{"showLegend":len(names)>1,"displayMode":"list","placement":"bottom"},
                 "tooltip":{"mode":"multi","sort":"none"}}})

# Qwen's vLLM instances are Prometheus job "vllm" (prometheus.yml). Gemma runs on llama.cpp
# (llamacpp: metrics, job "gemma"), but the Qwen panels select job="vllm" with J anyway.
def J(job="vllm",extra=""): return "{"+f'job="{job}"'+(","+extra if extra else "")+"}"
def V(m,w=RI,job="vllm"): return f"sum(rate(vllm:{m}{J(job)}[{w}]))"
TPOT=V("request_time_per_output_token_seconds_count")+" / "+V("request_time_per_output_token_seconds_sum")
TPOT5=V("request_time_per_output_token_seconds_count","5m")+" / "+V("request_time_per_output_token_seconds_sum","5m")
def q(p,m,w=RI,job="vllm"): return f"histogram_quantile({p}, sum by (le) (rate(vllm:{m}_bucket{J(job)}[{w}])))"
ACC=lambda w,job="vllm": f'1 + {V("spec_decode_num_accepted_tokens_total",w,job)} / {V("spec_decode_num_drafts_total",w,job)}'
HIT=lambda w,job="vllm": f'{V("prefix_cache_hits_total",w,job)} / {V("prefix_cache_queries_total",w,job)}'

# A card counts as busy while it reads more than 5 GB/s: idle cards read ~0-10 MB/s,
# decode and prefill tens of GB/s.
BUSY=lambda dev: f'avg_over_time((max(xpu_memory_read_bytes_per_second{{device="{dev}"}}) > bool 5e9)[1h:10s])'
RUN=('sum by (card) (vllm:num_requests_running) or sum by (card) (llamacpp:requests_processing)')

CARDS=["card 0","card 1"]
def per_card(metric): return f'max by (device) ({metric})'
# With PCIe ASPM on, the cards' idle power readings jump between ~0 and ~35 W from one
# sample to the next (B70-NOTES.md), so power is averaged. Readings under load are steady.
def POWER(w,dev=None):
    sel=f'{{device="{dev}"}}' if dev else ""
    return f'max by (device) (avg_over_time(xpu_power_watts{sel}[{w}]))'
PWR_NOTE=(" Averaged over {w}: with PCIe ASPM on, idle readings jump between ~0 and ~35 W, so idle "
          "figures are rough; use a wall meter for those. Readings under load are reliable.")

# Both cards together: what matters with TP=2 (one model on both) and for the whole box.
def total(*exprs): return " + ".join(f"({e} or vector(0))" for e in exprs)
GEN=lambda w: total(f"sum(rate(vllm:generation_tokens_total[{w}]))",f"sum(rate(llamacpp:tokens_predicted_total[{w}]))")
# llama.cpp's prompt_tokens_total counts only computed tokens (vLLM's includes cached ones).
PROMPT=lambda w: total(f"sum(rate(vllm:prompt_tokens_total[{w}]))",f"sum(rate(llamacpp:prompt_tokens_total[{w}]))",
                       f"sum(rate(llamacpp:prompt_tokens_cached_total[{w}]))")
GEN_BY_MODEL=(f"sum by (model) (rate(vllm:generation_tokens_total[{RI}])) or "
              f"sum by (model) (rate(llamacpp:tokens_predicted_total[{RI}]))")
MODELS=["Qwen3.8-27B","Qwen3.8-27B (TP=2)","gemma-4-31B","gpt-oss-120b"]

# --- Cluster: GPU layout and what runs on each card (layout_exporter.py) ---------------
# These panels get ids from 101 so the ids of the panels below stay as they were.
cluster=[];cid=[100]
def cnid(): cid[0]+=1; return cid[0]
GREY="#8e8e8e"
def name_stat(title,expr,legend,x,y,w,steps,desc="",mappings=None,noval="no data"):
    """Stat that shows each series' legend (a label value) coloured by its value."""
    cluster.append({"type":"stat","title":title,"description":desc,"id":cnid(),"datasource":DS,
      "gridPos":{"h":4,"w":w,"x":x,"y":y},
      "targets":[{"refId":"A","expr":expr,"legendFormat":legend,"instant":True,"range":False,"datasource":DS}],
      "fieldConfig":{"defaults":{"color":{"mode":"thresholds"},"thresholds":{"mode":"absolute","steps":steps},
        "mappings":mappings or [],"noValue":noval},"overrides":[]},
      "options":{"reduceOptions":{"calcs":["lastNotNull"],"fields":"","values":False},
        "colorMode":"background","graphMode":"none","textMode":"name","justifyMode":"center",
        "orientation":"vertical","wideLayout":True,"showPercentChange":False}})
# Value 1 = as configured / running, 0 = differs / free, -1 = unknown (inspect failed).
STATUS_STEPS=[{"color":CRIT,"value":None},{"color":WARN,"value":0},{"color":GOOD,"value":1}]
CARD_STEPS=[{"color":CRIT,"value":None},{"color":GREY,"value":0},{"color":C[0],"value":1}]
cy=0
cluster.append({"type":"row","title":"Cluster: GPU layout and model servers","id":cnid(),"collapsed":False,
                "gridPos":{"h":1,"w":24,"x":0,"y":cy},"panels":[]}); cy+=1
name_stat("Configured layout","local_serve_layout_info","{{layout}}",0,cy,4,[{"color":C[0],"value":None}],
  desc="The GPU layout `make mode` picked (gateway/llama-swap/layouts/mode): split = one Qwen per card, "
       "tp2 = one Qwen across both cards, review = Qwen on card 0 and Gemma on card 1, "
       "big = gpt-oss-120b across both cards.")
name_stat("Running layout","local_serve_effective_layout","{{layout}}",4,cy,5,STATUS_STEPS,
  desc="What the running containers add up to. Green: the configured layout. Amber: something else, "
       "e.g. during a layout switch, while qwen-image holds card 1 (\"+qwen-image\"), or \"mixed\" "
       "when no layout fits (a container still starting or crashed). Red: the Docker API did not answer.")
name_stat("Card 0 (PCI 03:00.0)",'local_serve_card_container{card="0"}',"{{container}}  {{model}}",9,cy,6,CARD_STEPS,
  desc="Model container(s) running on card 0 and the model each serves. Grey = free. A container counts "
       "from the moment it starts, before its model has loaded.")
name_stat("Card 1 (PCI 08:00.0)",'local_serve_card_container{card="1"}',"{{container}}  {{model}}",15,cy,6,CARD_STEPS,
  desc="Model container(s) running on card 1 and the model each serves. Grey = free. A container counts "
       "from the moment it starts, before its model has loaded.")
name_stat("Layout exporter","local_serve_docker_api_up","Docker API",21,cy,3,
  [{"color":CRIT,"value":None},{"color":GOOD,"value":1}],
  desc="Whether layout-exporter could inspect every model container through the docker-api proxy in its "
       "last poll. Red with no data: the exporter itself is down.",
  mappings=[{"type":"value","options":{"0":{"text":"down"},"1":{"text":"ok"}}}],noval="exporter down")
cluster[-1]["options"]["textMode"]="value_and_name"
cy+=4
# Running model servers with their launch config, health joined from the state metric.
SERVERS=("(local_serve_container_running == 1) * on (container) group_left (state, health) "
         "local_serve_container_state")
cols=["container","cards","model","engine","tp","ctx","parallel","quant","kv","spec","weights","health"]
names={"container":"Container","cards":"Card(s)","model":"Model","engine":"Engine","tp":"TP","ctx":"Context",
       "parallel":"Slots / max seqs","quant":"Quant","kv":"KV cache dtype","spec":"Speculative","weights":"Weights",
       "health":"Health"}
cluster.append({"type":"table","title":"Model servers running","id":cnid(),"datasource":DS,
  "description":"Each running model container's serving config, read from its launch command (docker inspect): "
    "TP = --tensor-parallel-size; Context = --max-model-len (vLLM) or --ctx-size (llama.cpp, shared by the slots); "
    "Slots / max seqs = --max-num-seqs (vLLM) or --parallel (llama.cpp); Quant = --quantization, else read from "
    "the weights' name; KV cache dtype = --kv-cache-dtype (vLLM, auto = the model's dtype) or "
    "--cache-type-k/-v (llama.cpp). Empty = not set or not applicable. Health = the container's healthcheck "
    "(starting while the model loads).",
  "gridPos":{"h":6,"w":24,"x":0,"y":cy},
  "targets":[{"refId":"A","expr":SERVERS,"instant":True,"range":False,"format":"table","datasource":DS}],
  "fieldConfig":{"defaults":{"custom":{"align":"auto","cellOptions":{"type":"auto"}}},"overrides":[
     {"matcher":{"id":"byName","options":"Health"},"properties":[
       {"id":"custom.cellOptions","value":{"type":"color-text"}},
       {"id":"mappings","value":[{"type":"value","options":{"healthy":{"color":GOOD},"starting":{"color":WARN},
                                                            "unhealthy":{"color":CRIT}}}]}]}]},
  "options":{"showHeader":True,"cellHeight":"sm","footer":{"show":False}},
  "transformations":[{"id":"organize","options":{
     "excludeByName":{k:True for k in ("Time","Value","__name__","job","instance","state")},
     "indexByName":{c:i for i,c in enumerate(cols)},"renameByName":names}}]})
cy+=6
panels.extend(cluster)

y=cy; row("Both GPUs combined",y); y+=1
stat("Generated tokens/s (5 min)",GEN("5m"),"none",0,y,w=6,decimals=0,
     desc="Tokens generated per second across every model on both cards, averaged over 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("Prompt tokens/s (5 min)",PROMPT("5m"),"none",6,y,w=6,decimals=0,
     desc="Prompt tokens read per second across every model on both cards (including tokens served from cache), averaged over 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("Total power (5 min)",f"sum({POWER('5m')})","watt",12,y,w=6,decimals=0,desc="Both B70s together."+PWR_NOTE.format(w="5 minutes"))
stat("Combined memory bandwidth","sum(max by (device) (xpu_memory_read_bytes_per_second))","Bps",18,y,w=6,decimals=0,
     desc="Memory read bandwidth of both B70s together.")
y+=4
ts("Generated tokens/s by model (stacked)",[(GEN_BY_MODEL,"{{model}}")],"suffix: tok/s",0,y,w=8,series=MODELS,stack=True,
   desc="Each model's generation rate, stacked: the top edge is the total across both cards. TP=2 shows as one model on both cards.")
ts("Power, both cards (stacked)",[(POWER("1m"),"card {{device}}")],"watt",8,y,w=8,series=CARDS,stack=True,
   desc="Each card's power, stacked: the top edge is the total draw of both B70s."+PWR_NOTE.format(w="1 minute"))
ts("Memory bandwidth, both cards (stacked)",[(per_card("xpu_memory_read_bytes_per_second"),"card {{device}}")],"Bps",16,y,w=8,series=CARDS,stack=True,
   desc="Each card's memory read bandwidth, stacked: the top edge is the combined bandwidth.")
y+=8

row("GPUs: card 0 vs card 1",y); y+=1
stat("Card 0 busy (last hour)",BUSY("0"),"percentunit",0,y,w=6,decimals=0,
     desc="Share of the last hour card 0 (PCI 03:00.0) was reading more than 5 GB/s from memory, i.e. running prefill or decode.")
stat("Card 1 busy (last hour)",BUSY("1"),"percentunit",6,y,w=6,decimals=0,color=C[1],
     desc="Share of the last hour card 1 (PCI 08:00.0) was reading more than 5 GB/s from memory, i.e. running prefill or decode.")
stat("Card 0 power (5 min)",POWER("5m","0"),"watt",12,y,w=6,decimals=0,desc=PWR_NOTE.format(w="5 minutes").strip())
stat("Card 1 power (5 min)",POWER("5m","1"),"watt",18,y,w=6,decimals=0,color=C[1],desc=PWR_NOTE.format(w="5 minutes").strip())
y+=4
ts("GPU activity: memory read bandwidth",[(per_card("xpu_memory_read_bytes_per_second"),"card {{device}}")],"Bps",0,y,w=12,series=CARDS,
   desc="Best proxy for GPU load: decode is limited by memory bandwidth and prefill reads the weights too. Near zero means the card is idle.")
ts("Requests running per card",[(RUN,"card {{card}}")],"none",12,y,w=12,decimals=0,series=["card 0","card 1","card 0+1"],
   desc="Requests each model server is working on, by the card(s) it runs on (vLLM, and llama.cpp for Gemma and gpt-oss). 0+1 is Qwen with TP=2 or gpt-oss.")
y+=8
ts("Power per card",[(POWER("1m"),"card {{device}}")],"watt",0,y,w=12,series=CARDS,desc=PWR_NOTE.format(w="1 minute").strip())
ts("VRAM used per card",[(per_card("xpu_memory_used_bytes"),"card {{device}}")],"bytes",12,y,w=12,series=CARDS)
y+=8

row("Qwen (vLLM): overview, last 5 minutes",y); y+=1
stat("Decode speed","%s"%TPOT5,"none",0,y,desc="Average per-request generation speed (tokens/s) over requests finished in the last 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("First token (median)",q(0.5,"time_to_first_token_seconds","5m"),"s",4,y,decimals=None,
     desc="Median time from request arrival to first streamed token. Dominated by prompt processing for long agent contexts.")
stat("MTP tokens per step",ACC("5m"),"none",8,y,decimals=2,desc="Mean tokens produced per decoding step with MTP speculative decoding (1 = no speedup, 5 = max with 4 draft tokens).")
stat("Prefix cache hit rate",HIT("5m"),"percentunit",12,y,decimals=0,desc="Share of prompt tokens served from the prefix cache instead of being recomputed.")
stat("KV cache usage",f"max(vllm:kv_cache_usage_perc{J()})","percentunit",16,y,decimals=0,desc="Share of the KV cache currently holding active requests.",
     thresholds=[{"color":GOOD,"value":None},{"color":WARN,"value":0.8},{"color":CRIT,"value":0.95}])
stat("Requests running",f"sum(vllm:num_requests_running{J()})","none",20,y,decimals=0,desc="Requests vLLM is processing, across all Qwen instances (4 per card, 8 with TP=2).")
y+=4

row("Qwen (vLLM): throughput",y); y+=1
ts("Token throughput (all requests)",[(V("generation_tokens_total"),"generated"),
    (V("prompt_tokens_total")+" - "+V("prompt_tokens_cached_total"),"prompt (computed)"),
    (V("prompt_tokens_cached_total"),"prompt (from cache)")],"suffix: tok/s",0,y,
    desc="Tokens per second across all requests. Prompt tokens split into freshly computed vs reused from the prefix cache.")
ts("Per-request decode speed",[(TPOT,"decode speed")],"suffix: tok/s",12,y,
    desc="Average generation speed of individual requests finishing in each interval (1 / time per output token).")
y+=8

row("Qwen (vLLM): latency",y); y+=1
ts("Time to first token",[(q(0.5,"time_to_first_token_seconds"),"p50"),(q(0.95,"time_to_first_token_seconds"),"p95")],"s",0,y)
ts("Request duration",[(q(0.5,"e2e_request_latency_seconds"),"p50"),(q(0.95,"e2e_request_latency_seconds"),"p95")],"s",12,y,decimals=1,
    desc="End-to-end request time, including thinking and the full answer.")
y+=8
ts("Prefill vs decode time per request (median)",[(q(0.5,"request_prefill_time_seconds"),"prefill"),(q(0.5,"request_decode_time_seconds"),"decode")],"s",0,y,
    desc="Where request time goes: processing the prompt vs generating the answer.")
ts("Prompt size per request",[(q(0.5,"request_prompt_tokens"),"p50"),(q(0.95,"request_prompt_tokens"),"p95")],"none",12,y,decimals=0,
    desc="Prompt length in tokens. Coding agents typically send 15-40k tokens.")
y+=8

row("Qwen (vLLM): speculative decoding (MTP) and caching",y); y+=1
ts("MTP acceptance by draft position",[(f"sum(rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{J(extra=f'position=\"{p}\"')}[{RI}])) / "+V("spec_decode_num_drafts_total"),f"draft {p+1}") for p in range(4)],
    "percentunit",0,y,maxv=1,decimals=0,desc="How often each drafted token is accepted. Code is typically 80-95%; prose falls off quickly by draft 3-4.")
ts("Cache",[(HIT(RI),"prefix cache hit rate"),(f"max(vllm:kv_cache_usage_perc{J()})","KV cache usage")],"percentunit",12,y,maxv=1,decimals=0)
y+=8

LT=lambda m,w=RI,job="gemma": f"sum(rate(llamacpp:{m}{J(job)}[{w}]))"
row("Gemma (llama.cpp, review layout)",y); y+=1
stat("Decode speed",f'{LT("tokens_predicted_total","5m")} / {LT("tokens_predicted_seconds_total","5m")}',"none",0,y,w=6,
     desc="Average generation speed while Gemma was generating, over the last 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("Prompt speed",f'{LT("prompt_tokens_total","5m")} / {LT("prompt_seconds_total","5m")}',"none",6,y,w=6,decimals=0,
     desc="Prompt tokens processed per second of prompt processing, over the last 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("MTP tokens per step",f'1 + {LT("spec_decode_num_accepted_tokens_total","5m")} / {LT("spec_decode_num_drafts_total","5m")}',"none",12,y,w=6,decimals=2,
     desc="Mean tokens produced per decoding step with Gemma's MTP draft head (1 = no speedup, 4 = max with 3 drafts).")
stat("Requests running",f"sum(llamacpp:requests_processing{J('gemma')})","none",18,y,w=6,decimals=0,desc="Requests Gemma is processing (2 slots).")
y+=4
ts("Gemma token throughput",[(LT("tokens_predicted_total"),"generated"),(LT("prompt_tokens_total"),"prompt")],"suffix: tok/s",0,y,
   desc="Tokens per second Gemma generated and read. Empty outside the review layout.")
ts("Gemma MTP acceptance by draft position",[(f'sum(rate(llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="{p}"}}[{RI}])) / '+LT("spec_decode_num_drafts_total"),f"draft {p+1}") for p in range(3)],
    "percentunit",12,y,maxv=1,decimals=0,desc="How often each token Gemma's MTP head drafts is accepted (3 drafts per step).")
y+=8

# gpt-oss's row goes here; its panels are added last (below) so the GPU detail panels keep their ids.
gy=y; y+=13

row("GPU detail",y); y+=1
ts("Memory bandwidth (read, write)",[(per_card("xpu_memory_read_bytes_per_second"),"read card {{device}}"),
   (per_card("xpu_memory_write_bytes_per_second"),"write card {{device}}")],"Bps",0,y,w=8,
   series=["read card 0","read card 1","write card 0","write card 1"])
ts("Core clock",[(per_card("xpu_frequency_mhz"),"card {{device}}")],"none",8,y,w=8,series=CARDS)
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: MHz"
ts("Temperature",[(per_card("xpu_temperature_celsius"),"card {{device}}")],"celsius",16,y,w=8,minv=None,series=CARDS)
y+=8
stat("Qwen requests completed (24h)",f"round(sum(increase(vllm:request_success_total{J()}[24h])))","none",0,y,w=6,decimals=0,
     desc="Qwen requests that finished in the last 24 hours, across all vLLM instances. llama.cpp (Gemma, gpt-oss) "
          "has no completed-requests counter; see the tokens panel next to this one.")
stat("Tokens generated (24h), all models",total("sum(increase(vllm:generation_tokens_total[24h]))",
     "sum(increase(llamacpp:tokens_predicted_total[24h]))"),"short",6,y,w=6,decimals=1,
     desc="Tokens generated in the last 24 hours by every model: Qwen, Gemma and gpt-oss.")
y+=4

GO=lambda m,w=RI: LT(m,w,"gpt-oss")
row("gpt-oss-120b (llama.cpp, big layout, both cards)",gy); gy+=1
stat("Decode speed",f'{GO("tokens_predicted_total","5m")} / {GO("tokens_predicted_seconds_total","5m")}',"none",0,gy,w=6,
     desc="Average generation speed while gpt-oss was generating, over the last 5 minutes. Empty outside the big layout.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("Prompt speed",f'{GO("prompt_tokens_total","5m")} / {GO("prompt_seconds_total","5m")}',"none",6,gy,w=6,decimals=0,
     desc="Prompt tokens processed per second of prompt processing, over the last 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("Requests running",f"sum(llamacpp:requests_processing{J('gpt-oss')})","none",12,gy,w=6,decimals=0,
     desc="Requests gpt-oss is processing (4 slots).")
stat("Requests waiting",f"sum(llamacpp:requests_deferred{J('gpt-oss')})","none",18,gy,w=6,decimals=0,
     desc="Requests queued for a free slot. More than 4 at once wait here.",
     thresholds=[{"color":GOOD,"value":None},{"color":WARN,"value":1}])
gy+=4
ts("gpt-oss token throughput",[(GO("tokens_predicted_total"),"generated"),
    (GO("prompt_tokens_total"),"prompt (computed)"),
    (GO("prompt_tokens_cached_total"),"prompt (from cache)")],"suffix: tok/s",0,gy,
   desc="Tokens per second gpt-oss generated and read, prompt split into computed vs reused from the slot cache. Empty outside the big layout.")
ts("gpt-oss requests",[(f"sum(llamacpp:requests_processing{J('gpt-oss')})","running"),
    (f"sum(llamacpp:requests_deferred{J('gpt-oss')})","waiting")],"none",12,gy,decimals=0,
   desc="Requests in its 4 slots and requests queued for one.")

dash={"title":"Local LLMs on 2x Arc Pro B70","uid":"qwen-vllm","editable":True,"graphTooltip":1,
      "refresh":"10s","time":{"from":"now-1h","to":"now"},"schemaVersion":41,"tags":["vllm","qwen","gemma","gpt-oss","layout"],"panels":panels,
      "timepicker":{"refresh_intervals":["5s","10s","30s","1m","5m"]}}
json.dump(dash,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"grafana/dashboards/vllm.json"),"w"),indent=2)
print(len(panels),"panels")
