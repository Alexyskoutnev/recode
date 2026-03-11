"""Base agent interface for task execution."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of running an agent on a single task."""

    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    raw_output: str = ""


class BaseAgent(ABC):
    """Abstract base for all agent backends (Claude Code, Gemini CLI, Codex)."""

    def __init__(
        self,
        system_prompt: str | None = None,
        max_turns: int = 10,
        model: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_turns = max_turns
        self._model = model

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a professional assistant completing real-world work tasks. "
            "Follow the instructions precisely. Produce the exact deliverables requested. "
            "IMPORTANT: Save ALL output files to the current working directory (./), "
            "never to /tmp/ or any other absolute path. Use relative paths only. "
            "If reference files are mentioned but not found in the working directory, "
            "use reasonable placeholder data and focus on demonstrating the correct "
            "methodology and structure. Be thorough but concise. Show your work."
        )

    @abstractmethod
    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        """Run the agent on a prompt in the given working directory."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    def _find_binary(self, npm_package: str, binary_name: str) -> str:
        """Find a CLI binary, preferring global install over npx."""
        path = shutil.which(binary_name)
        if path:
            return path
        # Fall back to npx
        return f"npx {npm_package}"

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int = 1800,
    ) -> tuple[str, str, int]:
        """Run a subprocess and return (stdout, stderr, returncode)."""
        logger.debug("[subprocess] %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", f"Process timed out after {timeout}s", -1

        return (
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            proc.returncode or 0,
        )

    @staticmethod
    def _parse_jsonl(raw: str) -> list[dict[str, Any]]:
        """Parse newline-delimited JSON (JSONL) output."""
        events = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
