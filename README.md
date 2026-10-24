![MemComposer](pic/memcomposer_banner.png)

### Learning to Assemble Memories for LLM Agents via Reinforcement Learning

![](pic/arxiv.svg) [Paper](https://arxiv.org/abs/2610.21091) · ![](pic/huggingface.svg) [Models & Data](https://huggingface.co/collections/Makrrr/memcomposer)

---

Agent memory lets an LLM agent reuse past experience without updating its weights. But a memory that
is similar to the current query is not necessarily useful, and how the memories are put together
matters. MemComposer reads the query and the retrieved candidate memories together. It then decides
which memories a frozen agent (the executor) receives, how many, and in what order. It is trained
with **ExPO** (Experience Policy Optimization), a GRPO-style objective. The reward is the task
success of the full agent interaction that the assembled memories lead to.



## Requirements

- **Platform:** Linux with CUDA GPUs. All experiments ran on a single node with 8×H100.
- **Training:** Python 3.12 and [verl](https://github.com/volcengine/verl), installed into `verl/`.
Read [`verl/README.md`](verl/README.md) before training. There are two environments:
  - **Qwen3-4B:** PyTorch 2.8, vLLM 0.10.2, transformers 4.56, peft 0.19, verl 0.7.1.
  - **Ministral-3-8B:** PyTorch 2.11, vLLM 0.24, transformers 5.9, peft 0.21, verl 0.9.1.
- **τ²-bench:** a separate environment with
[tau2-bench](https://github.com/sierra-research/tau2-bench) (`pip install -e`, `TAU2_ROOT` set to
the checkout). It also needs an OpenAI-compatible endpoint (`BEDROCK_BASE_URL`,
`BEDROCK_API_KEY`) for the GLM-4.7 user simulator and the hosted executors.
- **ALFWorld:** a separate environment with [ALFWorld](https://github.com/alfworld/alfworld). Run
`alfworld-download` and set `ALFWORLD_DATA`.
- **Memory banks:** an OpenAI-compatible embedding server for Qwen3-Embedding-8B, needed only to
build a bank's index or retrieve from it.

## Download the data and the adapters

```bash
hf download Makrrr/MemComposer-data --repo-type dataset --local-dir data \
    --exclude README.md --exclude .gitattributes
hf download Makrrr/MemComposer --local-dir checkpoints \
    --exclude README.md --exclude .gitattributes
```

The code reads `data/` through `MEMCOMPOSER_DATA`, which defaults to this repository's `data/`.
[`data/README.md`](data/README.md) and [`checkpoints/README.md`](checkpoints/README.md) describe the
contents.

## Quick start

### Use a memory bank

Each memory is a folder holding three files:

- `memory.json`: the full record;
- `digest.txt`: the compact form MemComposer reads;
- `full.txt`: the form the executor receives.

To build the MemComposer prompt for a query from its retrieved candidates:

```python
import sys; sys.path.insert(0, "memcomposer/tau2")
import memcomposer_prompt as mp

bank = mp.load_base("retail", root="data/tau2/memory_banks/ministral")   # one entry per memory
candidates = bank[:10]                                                    # e.g. a retrieval result
prompt = mp.build_prompt("Hi, I'd like to exchange a desk lamp from a recent order.",
                         candidates, domain="retail", template="ministral")
```

The reply holds an `<answer>` list of case numbers. Pass it to `parse_selection` to get the chosen
memories, in the order MemComposer gave them.

### Train

Start the simulation services for a domain, check that the health lines report that domain, then
launch the run:

```bash
training/tau2_ministral/launch_services.sh airline    # 4 tau2-bench services; executor on Bedrock
training/tau2_ministral/run_airline.sh
```

[`training/README.md`](training/README.md) covers every run, the hardware, the recipe and the data
format.

### Evaluate a released adapter

Every training script evaluates instead of training when it is given an adapter and both validation
switches:

```bash
VAL_ONLY=True VAL_BEFORE=True training/tau2_ministral/run_airline.sh \
    actor_rollout_ref.model.lora_adapter_path=$PWD/checkpoints/tau2_ministral_airline
```

## Build your own memory bank

- **τ²-bench:** [`memory_bank/tau2/`](memory_bank/tau2/README.md). Any agent in
`memory_bank/tau2/agents.py` can be used, and adding one takes a single line.
- **ALFWorld:** [`memory_bank/alfworld/`](memory_bank/alfworld/README.md).

## Citation

```bibtex
@misc{xie2026memcomposer,
  title         = {MemComposer: Learning to Assemble Memories for LLM Agents via Reinforcement Learning},
  author        = {Yixuan Xie and Xueqiang Xu and Xinyi Fan and Xinxi Lyu and Tao Feng and Xing Fan and
                   Chenlei Guo and Jie Hao and Shaowen Wang and Jiashuo Sun and Jiawei Han},
  year          = {2026},
  eprint        = {2610.21091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## License

- **Code and adapters:** [Apache-2.0](LICENSE).
- **Data:** CC-BY-4.0. It also contains material that keeps its own terms; see the
[dataset card](https://huggingface.co/datasets/Makrrr/MemComposer-data).

## Acknowledgements

This work builds on [τ²-bench](https://github.com/sierra-research/tau2-bench),
[ALFWorld](https://github.com/alfworld/alfworld), [verl](https://github.com/volcengine/verl) and
[vLLM](https://github.com/vllm-project/vllm).
