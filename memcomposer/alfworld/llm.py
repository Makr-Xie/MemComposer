#!/usr/bin/env python
"""OpenAI-compatible chat client for the ALFWorld executor.

One client covers both backends: a local vLLM server and Bedrock's OpenAI-compatible endpoint.
Bedrock credentials come from BEDROCK_BASE_URL / BEDROCK_API_KEY in the environment and are never
printed, logged or put in a result record.

default_chat() reads:
  ALFW_AGENT_MODEL  executor model id (default mistral.ministral-3-8b-instruct)
  ALFW_AGENT_BASE   base URL(s) of a local OpenAI-compatible server; unset = Bedrock
  ALFW_AGENT_KEY    API key for ALFW_AGENT_BASE (default 'local')
  ALFW_LLM_TIMEOUT  per-request timeout in seconds (default 120)

Uses stdlib urllib on purpose, so it imports in a minimal ALFWorld environment (textworld and
pyyaml only) without adding requests/openai there.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

def load_bedrock():
    """(base_url, api_key) from the environment. Raises if either is missing."""
    base, key = os.environ.get('BEDROCK_BASE_URL'), os.environ.get('BEDROCK_API_KEY')
    if not base or not key:
        raise RuntimeError('BEDROCK_BASE_URL / BEDROCK_API_KEY not set')
    return base.strip().rstrip('/'), key.strip()


class Chat:
    def __init__(self, model, base_url=None, api_key=None, temperature=0.0,
                 max_tokens=512, timeout=120, retries=4):
        if base_url is None:
            base_url, api_key = load_bedrock()
        # base_url may be a comma-separated list; calls round-robin over the endpoints so
        # several local vLLM servers share the load evenly.
        self.urls = [u.strip().rstrip('/') + '/chat/completions'
                     for u in str(base_url).split(',') if u.strip()]
        self.url = self.urls[0]
        self._rr = 0
        self.key = api_key or 'local'
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()   # one Chat is shared across episode threads
        self._tls = threading.local()   # usage of the calling thread's last request

    @property
    def last_usage(self):
        """`usage` of the most recent successful call made by THIS thread, or {} if none.

        The global counters above are shared by every episode thread; this is what lets one
        episode count its own tokens.
        """
        return getattr(self._tls, 'usage', {})

    def __call__(self, messages, temperature=None, max_tokens=None):
        body = json.dumps({
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature if temperature is None else temperature,
            'max_tokens': self.max_tokens if max_tokens is None else max_tokens,
        }).encode()
        last = None
        self._tls.usage = {}            # a failed call must not report the previous call's usage
        with self._lock:
            self._rr += 1
            url = self.urls[self._rr % len(self.urls)]
        for attempt in range(self.retries):
            req = urllib.request.Request(
                url, data=body,
                headers={'Content-Type': 'application/json',
                         'Authorization': f'Bearer {self.key}'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                u = out.get('usage') or {}
                self._tls.usage = u
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += u.get('prompt_tokens', 0)
                    self.completion_tokens += u.get('completion_tokens', 0)
                return out['choices'][0]['message'].get('content') or ''
            except Exception as e:
                last = e
                # Never echo the request: it carries the Authorization header.
                code = getattr(e, 'code', None)
                if code in (400, 401, 403, 404):
                    raise RuntimeError(f'LLM call rejected, HTTP {code}') from None
                time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f'LLM call failed after {self.retries} tries: {type(last).__name__}')


def default_chat(model=None, **kw):
    model = model or os.environ.get('ALFW_AGENT_MODEL', 'mistral.ministral-3-8b-instruct')
    base = os.environ.get('ALFW_AGENT_BASE')     # set this to use a local vLLM instead
    key = os.environ.get('ALFW_AGENT_KEY')
    # Long memory blocks queue behind each other on a busy vLLM server and can exceed the default
    # timeout. This only changes how long we wait for a reply, not what is generated.
    if os.environ.get('ALFW_LLM_TIMEOUT'):
        kw.setdefault('timeout', float(os.environ['ALFW_LLM_TIMEOUT']))
    if base:
        return Chat(model, base_url=base, api_key=key or 'local', **kw)
    return Chat(model, **kw)
