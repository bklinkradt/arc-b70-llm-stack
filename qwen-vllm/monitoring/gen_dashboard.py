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
def ts(title,targets,unit,x,y,w=12,h=8,desc="",minv=0,maxv=None,decimals=None):
    ov=[{"matcher":{"id":"byName","options":t[1]},"properties":[{"id":"color","value":{"mode":"fixed","fixedColor":C[i]}}]} for i,t in enumerate(targets)]
    d={"unit":unit,"min":minv,"color":{"mode":"fixed","fixedColor":C[0]},
       "custom":{"lineWidth":2,"fillOpacity":8,"showPoints":"never","spanNulls":30000,"axisSoftMin":0,"gradientMode":"none"}}
    if maxv is not None: d["max"]=maxv
    if decimals is not None: d["decimals"]=decimals
    panels.append({"type":"timeseries","title":title,"description":desc,"id":nid(),"datasource":DS,
      "gridPos":{"h":h,"w":w,"x":x,"y":y},
      "targets":[{"refId":chr(65+i),"expr":e,"legendFormat":l,"datasource":DS} for i,(e,l) in enumerate(targets)],
      "fieldConfig":{"defaults":d,"overrides":ov if len(targets)>1 else []},
      "options":{"legend":{"showLegend":len(targets)>1,"displayMode":"list","placement":"bottom"},
                 "tooltip":{"mode":"multi","sort":"none"}}})

TPOT=f"sum(rate(vllm:request_time_per_output_token_seconds_count[{RI}])) / sum(rate(vllm:request_time_per_output_token_seconds_sum[{RI}]))"
TPOT5="sum(rate(vllm:request_time_per_output_token_seconds_count[5m])) / sum(rate(vllm:request_time_per_output_token_seconds_sum[5m]))"
def q(p,m,w=RI): return f"histogram_quantile({p}, sum by (le) (rate(vllm:{m}_bucket[{w}])))"
ACC=lambda w: f"1 + sum(rate(vllm:spec_decode_num_accepted_tokens_total[{w}])) / sum(rate(vllm:spec_decode_num_drafts_total[{w}]))"
HIT=lambda w: f"sum(rate(vllm:prefix_cache_hits_total[{w}])) / sum(rate(vllm:prefix_cache_queries_total[{w}]))"

y=0; row("Overview (last 5 minutes)",y); y+=1
stat("Decode speed","%s"%TPOT5,"none",0,y,desc="Average per-request generation speed (tokens/s) over requests finished in the last 5 minutes.")
panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: tok/s"
stat("First token (median)",q(0.5,"time_to_first_token_seconds","5m"),"s",4,y,decimals=None,
     desc="Median time from request arrival to first streamed token. Dominated by prompt processing for long agent contexts.")
stat("MTP tokens per step",ACC("5m"),"none",8,y,decimals=2,desc="Mean tokens produced per decoding step with MTP speculative decoding (1 = no speedup, 5 = max with 4 draft tokens).")
stat("Prefix cache hit rate",HIT("5m"),"percentunit",12,y,decimals=0,desc="Share of prompt tokens served from the prefix cache instead of being recomputed.")
stat("KV cache usage","max(vllm:kv_cache_usage_perc)","percentunit",16,y,decimals=0,desc="Share of the KV cache currently holding active requests.",
     thresholds=[{"color":GOOD,"value":None},{"color":WARN,"value":0.8},{"color":CRIT,"value":0.95}])
stat("Requests running","sum(vllm:num_requests_running)","none",20,y,decimals=0,desc="Requests currently being processed (max 4).")
y+=4

row("Throughput",y); y+=1
ts("Token throughput (all requests)",[(f"sum(rate(vllm:generation_tokens_total[{RI}]))","generated"),
    (f"sum(rate(vllm:prompt_tokens_total[{RI}])) - sum(rate(vllm:prompt_tokens_cached_total[{RI}]))","prompt (computed)"),
    (f"sum(rate(vllm:prompt_tokens_cached_total[{RI}]))","prompt (from cache)")],"suffix: tok/s",0,y,
    desc="Tokens per second across all requests. Prompt tokens split into freshly computed vs reused from the prefix cache.")
ts("Per-request decode speed",[(TPOT,"decode speed")],"suffix: tok/s",12,y,
    desc="Average generation speed of individual requests finishing in each interval (1 / time per output token).")
y+=8

row("Latency",y); y+=1
ts("Time to first token",[(q(0.5,"time_to_first_token_seconds"),"p50"),(q(0.95,"time_to_first_token_seconds"),"p95")],"s",0,y)
ts("Request duration",[(q(0.5,"e2e_request_latency_seconds"),"p50"),(q(0.95,"e2e_request_latency_seconds"),"p95")],"s",12,y,decimals=1,
    desc="End-to-end request time, including thinking and the full answer.")
y+=8
ts("Prefill vs decode time per request (median)",[(q(0.5,"request_prefill_time_seconds"),"prefill"),(q(0.5,"request_decode_time_seconds"),"decode")],"s",0,y,
    desc="Where request time goes: processing the prompt vs generating the answer.")
ts("Prompt size per request",[(q(0.5,"request_prompt_tokens"),"p50"),(q(0.95,"request_prompt_tokens"),"p95")],"none",12,y,decimals=0,
    desc="Prompt length in tokens. Coding agents typically send 15-40k tokens.")
y+=8

row("Speculative decoding (MTP) and caching",y); y+=1
ts("MTP acceptance by draft position",[(f"sum(rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{{position=\"{p}\"}}[{RI}])) / sum(rate(vllm:spec_decode_num_drafts_total[{RI}]))",f"draft {p+1}") for p in range(4)],
    "percentunit",0,y,maxv=1,decimals=0,desc="How often each drafted token is accepted. Code is typically 80-95%; prose falls off quickly by draft 3-4.")
ts("Cache",[(HIT(RI),"prefix cache hit rate"),("max(vllm:kv_cache_usage_perc)","KV cache usage")],"percentunit",12,y,maxv=1,decimals=0)
y+=8

row("GPU: Intel Arc Pro B70",y); y+=1
stat("Power","max(xpu_power_watts)","watt",0,y,decimals=0)
stat("Temperature","max(xpu_temperature_celsius)","celsius",4,y,decimals=0,
     thresholds=[{"color":GOOD,"value":None},{"color":WARN,"value":80},{"color":CRIT,"value":90}])
stat("Core clock","max(xpu_frequency_mhz)","none",8,y,decimals=0); panels[-1]["fieldConfig"]["defaults"]["unit"]="suffix: MHz"
stat("Memory bandwidth","max(xpu_memory_read_bytes_per_second)","Bps",12,y,decimals=0,
     desc="Best proxy for GPU load: token generation is limited by memory bandwidth. The xe driver doesn't report a utilization %.")
stat("VRAM used","max(xpu_memory_used_bytes)","bytes",16,y,decimals=1)
stat("Requests completed (24h)","round(sum(increase(vllm:request_success_total[24h])))","none",20,y,decimals=0)
y+=4
ts("Memory bandwidth",[("max(xpu_memory_read_bytes_per_second)","read"),("max(xpu_memory_write_bytes_per_second)","write")],"Bps",0,y,w=8)
ts("Power",[("max(xpu_power_watts)","power")],"watt",8,y,w=8)
ts("Temperature",[("max(xpu_temperature_celsius)","core")],"celsius",16,y,w=8,minv=None)
y+=8

dash={"title":"Qwen3.8-27B on Arc Pro B70","uid":"qwen-vllm","editable":True,"graphTooltip":1,
      "refresh":"10s","time":{"from":"now-1h","to":"now"},"schemaVersion":41,"tags":["vllm","qwen"],"panels":panels,
      "timepicker":{"refresh_intervals":["5s","10s","30s","1m","5m"]}}
json.dump(dash,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"grafana/dashboards/vllm.json"),"w"),indent=2)
print(len(panels),"panels")
