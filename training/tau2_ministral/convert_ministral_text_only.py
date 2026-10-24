#!/usr/bin/env python
"""Turn Ministral-3-8B-Instruct-2512-BF16 into a text-only causal-LM repo verl can load.

    python convert_ministral_text_only.py --src <Ministral-3-8B-Instruct-2512-BF16 dir>   # dry run
    python convert_ministral_text_only.py --src <dir> --apply
        # -> <repo>/models/Ministral-3-8B-Instruct-2512-text (override with --dst)

Afterwards apply the tokenizer and RoPE fixes in verl/README.md to the output directory.

Why it is needed: the published repo is a `Mistral3ForConditionalGeneration` (vision tower,
projector and text model under one roof). verl builds its actor with `AutoModelForCausalLM`,
and that mapping refuses `Mistral3Config` -- only `Ministral3Config` (the inner text config) is
registered for causal LM, as `Ministral3ForCausalLM`. So the text half has to stand on its own.

Two things change, both mechanical:

  * every tensor keeps its values and loses the `language_model.` prefix
    (`language_model.model.layers.0...` -> `model.layers.0...`), which is what
    `Ministral3ForCausalLM` expects;
  * `vision_tower.*` (218 tensors) and `multi_modal_projector.*` (4) are dropped, and
    `config.json` becomes the inner `text_config` with `architectures: [Ministral3ForCausalLM]`.

Nothing is quantised, re-ordered or re-cast. The text weights are bit-identical to the ones in
the published checkpoint, so this is the same model minus the parts a text-only RL run never
touches -- but it is NOT the published checkpoint (PROVENANCE.md in the output says so).
"""
import argparse, json, shutil
from pathlib import Path

DST_DEFAULT = Path(__file__).resolve().parents[2] / 'models' / 'Ministral-3-8B-Instruct-2512-text'
KEEP = 'language_model.'
# Only tokenizer and generation files are copied; the processor/vision files have no meaning
# once the tower is gone.
COPY = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'generation_config.json', 'chat_template.jinja', 'tekken.json']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='local copy of mistralai/Ministral-3-8B-Instruct-2512-BF16')
    ap.add_argument('--dst', default=str(DST_DEFAULT))
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    SRC, DST = Path(a.src), Path(a.dst)

    from safetensors.torch import load_file, save_file

    idx = json.loads((SRC / 'model.safetensors.index.json').read_text())
    wm = idx['weight_map']
    keep = {k: v for k, v in wm.items() if k.startswith(KEEP)}
    drop = {k for k in wm if k not in keep}
    assert keep, 'no language_model.* tensors -- repo layout is not what this script expects'
    print(f'  keep {len(keep)} tensors, drop {len(drop)} '
          f'({len({k.split(".")[0] for k in drop})} top-level groups)')

    cfg = json.loads((SRC / 'config.json').read_text())
    text = dict(cfg['text_config'])
    text['architectures'] = ['Ministral3ForCausalLM']
    for k in ('dtype', 'torch_dtype'):
        if k in cfg and k not in text:
            text[k] = cfg[k]
    print(f'  config: model_type={text["model_type"]} layers={text["num_hidden_layers"]} '
          f'hidden={text["hidden_size"]} vocab={text["vocab_size"]}')

    shards = sorted({v for v in keep.values()})
    print(f'  rewriting {len(shards)} shards -> {DST}')
    if not a.apply:
        sample = list(keep)[:3]
        for s in sample:
            print(f'    {s}  ->  {s[len(KEEP):]}')
        print('\n  dry run; nothing written. --apply to do it.')
        return

    DST.mkdir(parents=True, exist_ok=True)
    new_map, total = {}, 0
    for shard in shards:
        src = load_file(str(SRC / shard))
        out = {k[len(KEEP):]: v for k, v in src.items() if k.startswith(KEEP)}
        if not out:
            continue
        save_file(out, str(DST / shard), metadata={'format': 'pt'})
        for k in out:
            new_map[k] = shard
        total += sum(v.numel() * v.element_size() for v in out.values())
        print(f'    {shard}: {len(out)} tensors')
        del src, out

    (DST / 'model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total}, 'weight_map': new_map}, indent=1))
    (DST / 'config.json').write_text(json.dumps(text, indent=2))
    for f in COPY:
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)
            print(f'    copied {f}')
    (DST / 'PROVENANCE.md').write_text(
        '# Ministral-3-8B-Instruct-2512, text tower only\n\n'
        'Built by `training/tau2_ministral/convert_ministral_text_only.py` from\n'
        '`mistralai/Ministral-3-8B-Instruct-2512-BF16`.\n\n'
        'The published repo is a `Mistral3ForConditionalGeneration` (vision tower + projector +\n'
        'text model). `AutoModelForCausalLM` rejects that config, so verl cannot build an actor\n'
        'from it. Here the `language_model.` prefix is stripped from every tensor and the vision\n'
        'tower and projector are dropped, leaving a plain `Ministral3ForCausalLM`.\n\n'
        f'- kept {len(keep)} tensors, dropped {len(drop)}\n'
        '- weight VALUES are untouched: no quantisation, no re-cast, no re-ordering\n'
        '- `config.json` is the published `text_config`, with architectures set to '
        '`Ministral3ForCausalLM`\n\n'
        '**This is not the published checkpoint.** Text weights are bit-identical, but a\n'
        'writeup should state that the vision tower was removed.\n')
    print(f'\n  done. total_size {total:,} bytes')


if __name__ == '__main__':
    main()
