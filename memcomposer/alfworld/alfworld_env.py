#!/usr/bin/env python
"""One ALFWorld game, addressed by its gamefile path.

Deliberately not built on `AlfredTWEnv`: that class scans a whole split's directory tree on
construction and then plays whatever it collected. The evaluator has to jump to an arbitrary
`task_id` on every request, so it registers the one game directly through `textworld.gym`.
Startup cost is then independent of split size.

`AlfredDemangler` is required: without it the observations keep TextWorld's mangled internal
object names instead of "cabinet 1".

Success is `info['won']`, read on every step and OR-ed, never only at the end (ALFWorld's own
evaluator takes the max over the episode), because an episode does not necessarily terminate on
the step that wins.
"""
import re
import threading

import fast_downward
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

TASK_RE = re.compile(r'Your task is to:\s*(.+?)\s*$', re.M)

# textworld is not thread-safe, and not only at load time: textworld.gym's env-id registry and
# the tatsu parser singletons used for game loading and for observation rendering are
# process-wide. Rendering runs on every step, so concurrent stepping corrupts the parser state and
# raises "IndexError: pop from empty list" mid-episode.
#
# So every call into textworld is serialised. This is cheap: a step takes milliseconds, while the
# LLM call between steps is far slower and runs outside the lock, so episodes still overlap.
# Reentrant so __init__ can hold it across make() and reset() while reset() takes it again.
_TW_LOCK = threading.RLock()

# fast_downward.load_lib() copies the 34.6 MB libdownward.so into a fresh TemporaryDirectory and
# dlopens it, once per game. The directory is deleted, but the mapping keeps the inode alive, so
# every game pins another copy on /tmp until the process exits. On a tmpfs /tmp this is RAM, and
# a long run eventually fills it.
#
# A private copy per live game is still required: the library keeps one problem in global state,
# and episodes interleave. But a closed game's copy can serve the next game, because PddlEnv only
# calls load_lib() in __init__ and every load()/reset() re-runs load_sas() on it. So copies are
# pooled: at most one per concurrently open game, instead of one per game ever played.
_ORIG_LOAD_LIB = fast_downward.load_lib
_LIB_POOL = []
_handed = []                    # libs handed out during the AlfworldGame construction in progress


def _pooled_load_lib():
    lib = _LIB_POOL.pop() if _LIB_POOL else _ORIG_LOAD_LIB()
    _handed.append(lib)
    return lib


fast_downward.load_lib = _pooled_load_lib


def _one(x):
    """textworld returns bare values for a single game and 1-element lists under the batch API."""
    return x[0] if isinstance(x, (list, tuple)) else x


def _cmds(x):
    """admissible_commands is list[str] for a single game, list[list[str]] under the batch API.

    Do NOT route this through _one(): for a single game that takes element 0, which is the first
    COMMAND, and listing a string yields its characters. The model then sees an action set of
    single letters and dutifully answers "o c a b i n e t 1".
    """
    if x and isinstance(x[0], (list, tuple)):
        return list(x[0])
    return list(x)


class AlfworldGame:
    def __init__(self, gamefile: str, max_steps: int = 50):
        self.gamefile = gamefile
        self.max_steps = max_steps
        infos = textworld.EnvInfos(won=True, admissible_commands=True)
        self._libs = []
        with _TW_LOCK:
            # Held across make() and the first reset() so the libs handed out here belong to
            # this game alone; close() returns them to the pool.
            _handed.clear()
            try:
                env_id = textworld.gym.register_game(gamefile, infos,
                                                     max_episode_steps=max_steps,
                                                     wrappers=[AlfredDemangler()])
                self._env = textworld.gym.make(env_id)
                self.reset()
            except BaseException:
                _LIB_POOL.extend(_handed)       # no game to close later; recycle now
                _handed.clear()
                raise
            self._libs = list(_handed)
            _handed.clear()

    def reset(self):
        with _TW_LOCK:
            obs, info = self._env.reset()
        self.steps = 0
        self.won = False
        self.done = False
        self.obs = self._clean(_one(obs))
        self.admissible = _cmds(info['admissible_commands'])
        self.won |= bool(_one(info['won']))
        m = TASK_RE.search(self.obs)
        # The goal line is part of the opening observation; the evaluator needs it separately to
        # build the prompt and to key retrieval on.
        self.task = m.group(1).strip() if m else ''
        return self.obs

    def step(self, action: str):
        with _TW_LOCK:
            obs, _reward, done, info = self._env.step(action)
        self.steps += 1
        self.obs = self._clean(_one(obs))
        self.admissible = _cmds(info['admissible_commands'])
        self.won |= bool(_one(info['won']))        # OR across the episode, never last-step only
        self.done = bool(_one(done)) or self.won or self.steps >= self.max_steps
        return self.obs, self.won, self.done

    @staticmethod
    def _clean(obs):
        obs = obs.strip()
        if obs.startswith('You arrive at loc '):
            obs = obs[obs.find('. ') + 2:]
        return obs

    def close(self):
        with _TW_LOCK:
            try:
                self._env.close()
            except Exception:
                pass
            # Idempotent: a second close() finds _libs empty and returns nothing twice.
            _LIB_POOL.extend(self._libs)
            self._libs = []
