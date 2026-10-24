#!/usr/bin/env python
"""Run one ALFWorld game with an LLM executor and an optional set of selected memories.

    run_one(task_id, chat, memories=[...], max_steps=20) -> {'won': 1, 'steps': 12, ...}

This is the ALFWorld counterpart of tau2's `run_one`: address a task by id, hand it the memories
somebody selected, play it out, return the outcome. Nothing here knows how the memories were
chosen, so the same function serves the no-memory baseline (memories=None), a fixed top-k
baseline, and a trained selector.

Tasks are looked up in the task index at ALFW_TASK_INDEX (default
$MEMCOMPOSER_DATA/alfworld/task_index.jsonl).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from alfworld_env import AlfworldGame                                    # noqa: E402
from alfworld_prompt import SYSTEM, build_prompt, parse_action           # noqa: E402

# built by memory_bank/alfworld/build_task_index.py
INDEX = Path(os.environ.get('ALFW_TASK_INDEX',
                            Path(os.environ.get('MEMCOMPOSER_DATA', HERE.parents[1] / 'data'))
                            / 'alfworld' / 'task_index.jsonl'))
_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        with INDEX.open() as f:
            for line in f:
                r = json.loads(line)
                _INDEX[r['task_id']] = r
    return _INDEX


def run_one(task_id, chat, memories=None, max_steps=20, history_length=10,
            keep_trajectory=True, temperature=0.0, max_format_retries=2):
    rec = index()[task_id]
    t0 = time.time()
    game = AlfworldGame(rec['gamefile'], max_steps=max_steps)
    history, traj = [], []
    status_counts = {'ok': 0, 'no_tag': 0, 'not_admissible': 0, 'no_think': 0}
    err = None
    hint = None
    retries = 0
    # The opening observation (room inventory + goal line) is the only place the full list of
    # receptacles appears; every later observation is local to where the agent stands.
    init_obs = game.obs
    won_step = 0 if game.won else None
    llm_calls, tok_in, tok_out = 0, 0, 0
    try:
        while not game.done:
            # What the model is shown at this call: the admissible list and any format hint.
            # Recorded so the prompt can be rebuilt and a suggested action can be checked
            # against what was actually available.
            shown = list(game.admissible)
            hint_shown = hint
            prompt = build_prompt(game.task, game.obs, game.admissible, history,
                                  memories=memories, history_length=history_length, hint=hint)
            reply = chat([{'role': 'system', 'content': SYSTEM},
                          {'role': 'user', 'content': prompt}], temperature=temperature)
            llm_calls += 1
            usage = getattr(chat, 'last_usage', {}) or {}
            tok_in += usage.get('prompt_tokens', 0)
            tok_out += usage.get('completion_tokens', 0)
            action, status, think = parse_action(reply, game.admissible)
            status_counts[status] += 1
            # Counted, never penalised: a missing <think> block still yields a usable action, and
            # the frozen executor is not being trained to format. SkillRL zeroes its validity flag
            # for this; we only want to see how often it happens.
            if not think:
                status_counts['no_think'] += 1
            if status != 'ok':
                # The environment is not advanced and history is left alone -- a formatting slip
                # is not something that happened in the room. The correction rides on the next
                # prompt as a one-shot hint, which also breaks the temperature-0 fixed point.
                # After max_format_retries the step is burned, so a model that cannot follow the
                # format still exhausts its budget rather than looping forever.
                traj.append({'step': game.steps + 1, 'attempt': retries + 1, 'think': think,
                             'raw': reply, 'action': action, 'status': status,
                             'hint': hint_shown, 'admissible': shown})
                retries += 1
                hint = ('Your last reply had no usable <action> tag.' if status == 'no_tag' else
                        f'`{action}` is not in the admissible list. Copy one entry exactly.')
                if retries > max_format_retries:
                    game.steps += 1
                    retries, hint = 0, None
                    if game.steps >= max_steps:
                        break
                continue
            attempt = retries + 1
            obs, won, done = game.step(action)
            retries, hint = 0, None
            history.append((action, obs))
            if won and won_step is None:
                won_step = game.steps
            traj.append({'step': game.steps, 'attempt': attempt, 'think': think,
                         'action': action, 'status': 'ok', 'obs': obs, 'won': won,
                         'raw': reply, 'hint': hint_shown, 'admissible': shown})
            if done:
                break
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    finally:
        game.close()

    return {
        'task_id': task_id,
        'group_id': rec['group_id'],
        'task_type': rec['task_type'],
        'task': game.task,
        # Task metadata copied from the index so a record stands on its own.
        'split': rec.get('split'),
        'gamefile': rec.get('gamefile'),
        'floor_plan': rec.get('floor_plan'),
        'object_target': rec.get('object_target'),
        'parent_target': rec.get('parent_target'),
        'mrecep_target': rec.get('mrecep_target'),
        'toggle_target': rec.get('toggle_target'),
        'object_sliced': rec.get('object_sliced'),
        'expert_plan_len': rec.get('expert_plan_len'),
        'init_obs': init_obs,
        'won': int(game.won),
        'won_step': won_step,
        'end_reason': 'won' if game.won else ('error' if err else 'max_steps'),
        'reward': float(game.won),
        # `steps` includes steps burned by repeated format failures; `env_actions` counts only
        # actions the environment actually executed.
        'steps': game.steps,
        'env_actions': len(history),
        'max_steps': max_steps,
        'llm_calls': llm_calls,
        'prompt_tokens': tok_in,
        'completion_tokens': tok_out,
        'n_memories': len(memories or []),
        'format': status_counts,
        # Share of LLM calls that produced a usable admissible action, SkillRL's
        # `episode/valid_action_ratio`. Denominator is calls, not steps, so retries count against
        # it -- that is the point, it measures how much of the budget the format cost us.
        'valid_action_ratio': round(
            status_counts['ok'] / max(1, sum(status_counts[k] for k in
                                             ('ok', 'no_tag', 'not_admissible'))), 4),
        'error': err,
        'secs': round(time.time() - t0, 1),
        'trajectory': traj if keep_trajectory else None,
    }


def main():
    import argparse
    from llm import default_chat
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-ids', nargs='+', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    chat = default_chat(a.model, max_tokens=a.max_tokens)
    print(f'executor {chat.model}  max_steps {a.max_steps}', flush=True)
    rows = []
    for tid in a.task_ids:
        r = run_one(tid, chat, memories=None, max_steps=a.max_steps)
        rows.append(r)
        print(f"  {'WON ' if r['won'] else 'lost'} {r['steps']:3d} steps  {r['secs']:6.1f}s  "
              f"fmt {r['format']}  {tid.split('/')[1][:48]}", flush=True)
        if r['error']:
            print(f'    error: {r["error"]}', flush=True)
    n = len(rows)
    print(f"\nsuccess {sum(r['won'] for r in rows)}/{n}  "
          f"mean steps {sum(r['steps'] for r in rows)/n:.1f}  "
          f"LLM calls {chat.calls}  tokens in/out {chat.prompt_tokens}/{chat.completion_tokens}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'-> {a.out}')


if __name__ == '__main__':
    main()
