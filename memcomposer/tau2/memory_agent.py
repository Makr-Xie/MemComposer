"""Agent that carries a pre-rendered <experience> block, for tau2-bench.

tau2's LLMAgent builds its system prompt as <instructions> + <policy>. This adds <experience>
as a THIRD sibling. It is deliberately not concatenated into domain_policy: inside <policy> the
cases would inherit the policy's authority, and many memories describe behaviour that must not
be copied.

tau2_runner passes the rendered block directly (`experience=`). Through tau2's own CLI,
create_memory_llm_agent instead looks it up by task id in a pre-rendered `blocks.json`, so no
embedding server is needed at run time; the directory holding it is named by the MEMSEL_BLOCKS
env var, because tau2's CLI has no channel for custom agent arguments.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent
from tau2.environment.tool import Tool

SYSTEM_PROMPT_WITH_EXPERIENCE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
{experience}
""".strip()

_CACHE: dict[str, dict[str, str]] = {}


def load_blocks(path: Optional[str]) -> dict[str, str]:
    """task_id -> block. Empty dict when unset, which yields the stock tau2 prompt."""
    if not path:
        return {}
    if path not in _CACHE:
        blocks = Path(path) / 'blocks.json'
        if not blocks.exists():
            raise FileNotFoundError(f'MEMSEL_BLOCKS={path} has no blocks.json')
        _CACHE[path] = json.loads(blocks.read_text())
    return _CACHE[path]


class MemoryLLMAgent(LLMAgent):
    def __init__(self, tools: List[Tool], domain_policy: str, llm: str,
                 llm_args: Optional[dict] = None, experience: str = ""):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.experience = (experience or "").strip()

    @property
    def system_prompt(self) -> str:
        if not self.experience:
            return super().system_prompt
        return SYSTEM_PROMPT_WITH_EXPERIENCE.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=self.domain_policy,
            experience=self.experience,
        )


def create_memory_llm_agent(tools, domain_policy, **kwargs):
    """Factory matching tau2's create_llm_agent signature.

    build_agent passes task=task, which is how the per-task block is selected.
    """
    task = kwargs.get("task")
    blocks = load_blocks(os.environ.get("MEMSEL_BLOCKS"))
    experience = blocks.get(str(task.id), "") if (task is not None and blocks) else ""
    # Guard against a task the block set forgot. An empty block is NOT that: a selector with a
    # free choice of k may select nothing, and "" is the documented no-memory fallback.
    if blocks and task is not None and str(task.id) not in blocks:
        raise KeyError(f'no experience block for task {task.id} in {os.environ["MEMSEL_BLOCKS"]}')
    return MemoryLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        experience=experience,
    )
