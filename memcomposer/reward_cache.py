"""Append-only reward cache, keyed by everything that determines a simulation's inputs.

    from reward_cache import RewardCache
    c = RewardCache('out/reward_cache.jsonl')
    c.get('15', opening, ['task12_trial6'])     # -> [rewards...] or []
    c.put('15', opening, ['task12_trial6'], 1.0, seed=0)

Two rollouts in a GRPO group that pick the same memories in the same order must receive the
same reward: otherwise the noise between two identical actions becomes a spurious advantage.
Order is part of the key because the executor sees the cases in the selected order. The
opening is hashed rather than stored inline to keep keys short. Enabled by MEMSEL_REWARD_CACHE.
"""
import hashlib, json, threading
from pathlib import Path


def make_key(task_id, opening, folders):
    h = hashlib.sha1(opening.strip().encode()).hexdigest()[:12]
    return f"{task_id}|{h}|{','.join(folders)}"


class RewardCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._d = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._d.setdefault(r['key'], []).append(r['reward'])

    def get(self, task_id, opening, folders):
        return list(self._d.get(make_key(task_id, opening, folders), []))

    def put(self, task_id, opening, folders, reward, **extra):
        key = make_key(task_id, opening, folders)
        rec = {'key': key, 'task_id': str(task_id), 'n_folders': len(folders),
               'reward': float(reward), **extra}
        with self._lock:
            self._d.setdefault(key, []).append(float(reward))
            with self.path.open('a') as f:
                f.write(json.dumps(rec) + '\n')

    def __len__(self):
        return len(self._d)
