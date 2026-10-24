# verl

Install [verl](https://github.com/volcengine/verl) here; the training scripts run from this directory.

```bash
git clone https://github.com/volcengine/verl.git verl/   # then install it into your training environment
```

Two things go wrong **silently** if they are not handled: the run trains, the curves look normal, and
the model is wrong. Both are described below with the exact fixes we used.

---

## 1. The LoRA adapter must actually reach vLLM

**Symptom.** With `actor.strategy=fsdp` (FSDP1) and `peft >= 0.21`, every weight sync collects **zero**
LoRA tensors. verl does not raise an error, and vLLM keeps generating with the base model. The reward stays
flat, and every "trained" evaluation is really the base model.

**Cause.** FSDP1's `state_dict` hooks strip `_fsdp_wrapped_module.` from parameter keys, while
`named_modules()` keeps it. peft builds its adapter prefixes from `named_modules()`, so nothing matches.

**Fix.** Use one of these combinations:

| run | strategy | peft |
|---|---|---|
| Qwen3-4B selector (τ²-bench) | `fsdp` | `0.19.1` |
| Ministral-3-8B selector (τ²-bench, ALFWorld) | `fsdp2` | `0.21.x` |

**Guards (recommended).** These turn the silent failure into an error. Use the first in FSDP1 workers
(`verl/workers/fsdp_workers.py`, where the LoRA params for the rollout are collected):

```python
if self.base_sync_done and not any("lora_" in k for k in params):
    raise RuntimeError(
        f"LoRA weight sync collected 0 adapter tensors from {len(params)} keys at this step, so vLLM "
        "would keep serving the previous (base) weights. Known cause: peft >= 0.21 with FSDP1 -- pin "
        "peft==0.19.1 or switch to strategy=fsdp2.")
```

Use the second in the FSDP engine (`verl/workers/engine/fsdp/transformer_impl.py`, after
`collect_lora_params(...)`):

```python
if base_sync_done and not params:
    raise RuntimeError("LoRA weight sync collected 0 adapter tensors from the actor; "
                       "use strategy=fsdp2, or pin peft==0.19.1 with FSDP1.")
```

Use the third on the vLLM side (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`, in
`generate`, where the `LoRARequest` is built). Upstream silently falls back to the base model when the
adapter is not registered:

```python
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )
            else:                                   # added: upstream has no else branch
                raise RuntimeError(
                    f"LoRA adapter {VLLM_LORA_INT_ID} is not loaded in the vLLM engine "
                    f"(list_loras={await self.engine.list_loras()}). Refusing to generate from the base model.")
```

---

## 2. Ministral-3-8B needs four fixes before it trains correctly

We saw these with transformers 5.9 and vLLM 0.24. Each one is silent. Together they produce this
signature, which means a pipeline bug, not a model property:

| metric | broken | healthy |
|---|---|---|
| reward | ≈ 0 | rising |
| `actor/entropy` | ≈ 1.6 | 0.1 – 0.4 |
| `training/rollout_probs_diff_mean` | ≈ 0.25 | ≈ 0.003 |

### (0) Use the text tower only

The published `mistralai/Ministral-3-8B-Instruct-2512-BF16` is a `Mistral3ForConditionalGeneration`
(vision tower + projector + text model), and verl cannot build an actor from it. Strip the
`language_model.` prefix from the text tensors, drop the vision tower and projector, and save the
published `text_config` with `architectures: ["Ministral3ForCausalLM"]`. The weight values are
unchanged. The conversion script is `training/tau2_ministral/convert_ministral_text_only.py`.

### (a) Tokenizer: wrong decoder, so every reward is 0

transformers 5.x maps `tokenizer_class: LlamaTokenizerFast` to a tokenizer with a SentencePiece decoder.
Tekken is byte-level BPE, so `decode()` returns `Ġ`/`Ċ` artifacts and no answer can be parsed. The
token ids are correct, which is why an id-level check does not catch this. In the model's
`tokenizer_config.json`:

```diff
-  "tokenizer_class": "LlamaTokenizerFast",
+  "tokenizer_class": "PreTrainedTokenizerFast",
+  "fix_mistral_regex": true
```

### (b) RoPE: vLLM applies a YaRN scale the model does not use

The official `params.json` has `yarn.apply_scale: false`. vLLM's HF-format YaRN path ignores `mscale` and
multiplies cos/sin by 1.277, so rollout and training log-probs disagree by about 0.04
([vllm#48621](https://github.com/vllm-project/vllm/issues/48621), fixed after 0.24). In the model's
`config.json`:

```diff
   "rope_parameters": {
     ...
+    "apply_yarn_scaling": false
   },
+  "llama_4_scaling": {"original_max_position_embeddings": 16384, "beta": 0.1},
+  "ignore_keys_at_rope_validation": ["apply_yarn_scaling", "llama_4_scaling_beta", "max_position_embeddings"],
```

### (c) Attention: packed sequences attend across each other

`Ministral3Attention.forward` takes `position_ids` as a named argument and does not pass it to the
attention interface. With `use_remove_padding=True`, flash-attn therefore never sees the sequence
boundaries, and every packed sequence after the first attends to the ones before it. This corrupts the
actor and ref log-probs and the policy gradient. Patch one line in the training environment's
`transformers/models/ministral3/modeling_ministral3.py`, in the `attention_interface(...)` call:

```diff
             sliding_window=getattr(self.config, "sliding_window", None),
+            position_ids=position_ids,
             **kwargs,
         )
```

Guard (optional), in `verl/models/transformers/monkey_patch.py` inside `apply_monkey_patch`, so that a
reinstall of transformers fails loudly instead of training on noise:

```python
if use_remove_padding and getattr(model.config, "model_type", None) == "ministral3":
    import inspect
    from transformers.models.ministral3.modeling_ministral3 import Ministral3Attention
    if "position_ids=position_ids" not in inspect.getsource(Ministral3Attention.forward):
        raise RuntimeError("Ministral3Attention does not forward position_ids; patch "
                           "modeling_ministral3.py or set use_remove_padding=False.")
```

---

## Also

- **Rollout dumps crash on numpy scalars** (`trainer.rollout_data_dir` set, older verl). In
  `RayPPOTrainer._dump_generations`, use `json.dumps(entry, ensure_ascii=False, default=str)`.
- **Evaluate the base model plus the adapter, not a merged model.** The LoRA update is about 0.1% of
  the base weights, which is below bf16 precision, so a bf16 merge erases most of it. In one
  ALFWorld comparison the merged model scored 0.77 and the base model plus the adapter scored 0.82.
  Some verl merger versions also write the base weights plus a separate `lora_adapter/` instead of
  merging. Diff the result against the base model before using it.
