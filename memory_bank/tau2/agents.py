#!/usr/bin/env python
"""Registry of the agents whose memory banks this pipeline builds.

Every stage of the pipeline -- rollouts, selection, review, top-up, rendering -- needs the
same four facts about an agent: the litellm model id, where its no-memory train rollouts landed,
where its memory bank lives, and which credentials reach it.

    from agents import AGENTS, creds
    a = AGENTS['qwen32b']
    a.rollouts   -> '<data>/rollouts/qwen32b/{}'     (.format(domain))
    a.base       -> Path('<data>/memory_banks/qwen32b')
    a.model      -> 'openai/qwen.qwen3-32b'
    creds(a)     -> {'OPENAI_API_KEY': ..., 'OPENAI_API_BASE': ...}

Credentials come from the environment only:
    bedrock  BEDROCK_API_KEY, BEDROCK_BASE_URL
    openai   OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1)
    local    OPENAI_API_KEY (default 'dummy'), OPENAI_API_BASE
"""
import os
from dataclasses import dataclass
from pathlib import Path

import paths


@dataclass(frozen=True)
class Agent:
    tag: str
    model: str          # litellm model id, as used by tau2 and the review scripts
    provider: str       # 'bedrock' | 'openai' | 'local'
    label: str          # model name written into provenance strings

    @property
    def rollouts(self):
        """Template for the no-memory train rollout directory; .format(domain)."""
        return str(paths.rollouts(self.tag, '{}'))

    @property
    def base(self):
        return paths.bank(self.tag)


AGENTS = {
    'ministral': Agent('ministral', 'openai/mistral.ministral-3-8b-instruct', 'bedrock',
                       'ministral-3-8b'),
    'qwen': Agent('qwen', 'openai/Qwen3-4B-Instruct-2507', 'local', 'qwen3-4b'),
    'qwen32b': Agent('qwen32b', 'openai/qwen.qwen3-32b', 'bedrock', 'qwen3-32b'),
    'qwen235b': Agent('qwen235b', 'openai/qwen.qwen3-235b-a22b-2507', 'bedrock',
                      'qwen3-235b-a22b-2507'),
    'gpt54mini': Agent('gpt54mini', 'openai/gpt-5.4-mini', 'openai', 'gpt-5.4-mini'),
    'gpt54': Agent('gpt54', 'openai/gpt-5.4', 'openai', 'gpt-5.4'),
}


def creds(agent):
    """The OPENAI_API_* pair that reaches this agent. Values are never logged."""
    e = os.environ
    if agent.provider == 'bedrock':
        k, b = e.get('BEDROCK_API_KEY'), e.get('BEDROCK_BASE_URL')
    elif agent.provider == 'openai':
        k, b = e.get('OPENAI_API_KEY'), e.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    else:
        k, b = e.get('OPENAI_API_KEY', 'dummy'), e.get('OPENAI_API_BASE', '')
    if not k:
        raise SystemExit(f'no credentials for provider {agent.provider!r}')
    return {'OPENAI_API_KEY': k, 'OPENAI_API_BASE': b, 'OPENAI_BASE_URL': b}


def add_agent_arg(ap):
    ap.add_argument('--agent', required=True, choices=sorted(AGENTS),
                    help='which agent this memory bank belongs to')
    return ap
