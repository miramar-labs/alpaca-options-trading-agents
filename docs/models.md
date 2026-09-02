# Model choice — the Dealer / Analyst LLM

Every LLM call in this system (the Analyst's universe screen, the Dealer's
BUY/HOLD/SELL signal, and the MCP contract-selection loop) runs against one
locally-hosted model, set by `llm.model` in [`config.yaml`](../config.yaml) and
served by **Ollama** on the DGX Spark (`http://DGX_HOST_IP:11434/v1`, an
OpenAI-compatible endpoint). No external LLM API is ever in the path.

Current model: **`qwen2.5:32b-instruct-q4_K_M`**.

## Why not a bigger / fancier model

The DGX Spark (GB10, 128 GB unified LPDDR5X, ~273 GB/s) is **memory-bandwidth
bound** for token generation, not compute bound. Generation speed is roughly
`bandwidth / bytes-read-per-token`, so a q4 32B dense model (~19 GB resident)
runs about 3x faster per token than a q4 70B and about the same speed as a much
larger MoE with a comparable active-parameter count — while fitting in memory
several times over.

## Why not the previous model (`qwen3.6:35b-a3b`)

The project originally ran `qwen3.6:35b-a3b` — a 35B mixture-of-experts model
with only ~3B active parameters per token. It is fast, but it **reliably
returned an empty completion** from the
`.with_structured_output(OptionContractPick)` call at the end of the MCP
contract-selection loop. Every one of the first seven option positions (1–2 Sep
2026) was therefore placed by the deterministic `_fallback_pick`, not the model.

Root cause: weak **constrained decoding** from the low active-parameter count —
not context length. The failing prompt was only ~4k tokens against a 32k
context window, and the model returned `content=''` with `finish_reason: stop`
after a single completion token. Smaller single-turn structured calls (the
Dealer's own `Signal`) mostly succeeded; the larger 7-field `OptionContractPick`
schema at the end of a tool-laden transcript did not.

## What was tested

Each candidate was run through the **actual** `_select_option_contract_async`
path on the DGX, against live Alpaca MCP option-chain data, for real underlyings
— the same check now committed as
[`scripts/check_contract_selection.py`](../scripts/check_contract_selection.py):

| Model | Resident | Gen speed | Structured pick | Notes |
|---|---|---|---|---|
| `qwen3.6:35b-a3b` (35B MoE, ~3B active) | 22 GB | fast | **empty → fallback** | the baseline failure |
| `gpt-oss:120b` (117B MoE, ~5B active) | 64 GB | fast | **empty → fallback** | harmony/reasoning format fights the structured call |
| `llama3.3:70b-instruct-q4_K_M` | 42 GB | ~4.8 tok/s | valid pick | works, but slow; generic reasoning text |
| `qwen2.5-coder:32b-instruct-fp16` | 73 GB | ~3.7 tok/s | valid pick | best reasoning text, but fp16 is too slow / too large |
| **`qwen2.5:32b-instruct-q4_K_M`** | **19 GB** | **~9.8 tok/s** | **valid pick** | **chosen** |

`qwen2.5:32b-instruct` is a dense, **non-reasoning** instruct model. That matters
here: there is no per-call "thinking" token tax on the high-frequency
BUY/HOLD/SELL signal, it has enough capacity for reliable constrained decoding,
and the q4 quant keeps it in the bandwidth sweet spot. `request_timeout_s` was
raised 120 → 180 for margin on the slightly slower per-call latency.

Ollama is configured with `OLLAMA_KEEP_ALIVE=30m` (systemd drop-in) so the model
stays resident between the Dealer's 10-minute poll cycles instead of cold-loading
each time.

## Verifying the live model

`scripts/check_contract_selection.py` runs the real selection path (MCP
subprocess, tool loop, `with_structured_output`) without touching the
signal / duplicate / risk gates, so it never places an order. Run it in the
dealer pod:

```
kubectl -n alpaca-options-trader exec deploy/dealer -- python /tmp/ccs.py NVDA BUY
```

It prints `fell_back=False` and the pick JSON when the model returns a valid
structured contract, `fell_back=True` when `_fallback_pick` had to salvage it.
Post-switch check on 2 Sep 2026, `qwen2.5:32b-instruct-q4_K_M`, all
`fell_back=False`:

| Symbol | Signal | Pick | Δ | Premium |
|---|---|---|---|---|
| NVDA | BUY | `NVDA260916C00230000` call | 0.35 | $3.08 |
| TSLA | BUY | `TSLA260916C00360000` call | 0.47 | $10.32 |
| AAPL | SELL | `AAPL260916P00325000` put | 0.48 | $6.10 |

Each pick is the correct right (call ↔ BUY, put ↔ SELL) and lands inside the
configured delta and DTE windows.

## Ollama vs. a guided-decoding backend (vLLM / SGLang / NIM)

A guided-decoding backend (outlines / xgrammar grammar-constrained sampling)
would make structured output essentially bulletproof regardless of model size,
and is the "correct" long-term answer. It was not adopted for this project
because it is an infrastructure migration off Ollama (ARM64 / GB10 image
availability, k8s serving templates, KV-cache tuning) that does not fit the
hackathon window. With a capable-enough dense model the Ollama path is reliable
in practice, and `_fallback_pick` remains as a first-class safety net for any
call that still comes back empty.
