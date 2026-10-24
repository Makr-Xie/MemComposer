# Checkpoints

The trained MemComposer selectors, as LoRA adapters (peft format, fp32): r = 64, α = 128, on all
linear layers (`q/k/v/o_proj`, `gate/up/down_proj`). Only this README is tracked in git; the
adapters are downloaded from Hugging Face.

| directory | benchmark | base model | training script |
|---|---|---|---|
| `tau2_qwen_retail` | τ²-bench retail | Qwen3-4B-Instruct-2507 | `training/tau2_qwen/run_retail.sh` |
| `tau2_qwen_telecom` | τ²-bench telecom | Qwen3-4B-Instruct-2507 | `training/tau2_qwen/run_telecom.sh` |
| `tau2_ministral_retail` | τ²-bench retail | Ministral-3-8B-Instruct-2512 | `training/tau2_ministral/run_retail.sh` |
| `tau2_ministral_airline` | τ²-bench airline | Ministral-3-8B-Instruct-2512 | `training/tau2_ministral/run_airline.sh` |
| `tau2_ministral_telecom` | τ²-bench telecom | Ministral-3-8B-Instruct-2512 | `training/tau2_ministral/run_telecom.sh` |
| `alfworld_ministral` | ALFWorld | Ministral-3-8B-Instruct-2512 | `training/alfworld/run_alfworld.sh` |

Each adapter holds exactly the LoRA weights of the training checkpoint evaluated in the paper.

## Base models

- **Qwen3-4B-Instruct-2507:** use `Qwen/Qwen3-4B-Instruct-2507` as published.
- **Ministral-3-8B-Instruct-2512:** use the text-only conversion of
  `mistralai/Ministral-3-8B-Instruct-2512-BF16`. Build it with
  `training/tau2_ministral/convert_ministral_text_only.py`, then apply the tokenizer and RoPE fixes
  in `verl/README.md`. The adapters' module names follow that conversion
  (`model.layers.<i>...`).

## Using an adapter

Run the base model with the adapter attached; do not merge it into bf16 weights. The LoRA update is
about 0.1% of the base weights, below bf16 precision, so a bf16 merge erases most of it.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype="auto")
model = PeftModel.from_pretrained(model, "checkpoints/tau2_qwen_retail")
```

With vLLM, serve the base model with `--enable-lora --max-lora-rank 64` and pass the adapter
directory as a LoRA module.
