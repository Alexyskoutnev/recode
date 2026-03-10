"""OpenAI Codex CLI agent backend — runs tasks via the Codex CLI."""

from __future__ import annotations

import logging
from pathlib import Path

from src.eval.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class CodexAgent(BaseAgent):
    """Runs tasks through OpenAI Codex CLI (npx @openai/codex)."""

    def name(self) -> str:
        return "codex"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        bin_path = self._find_binary("@openai/codex", "codex")

        # Ensure cwd is absolute (Codex needs it to exist)
        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        # Init a git repo if needed (Codex requires it)
        git_dir = cwd / ".git"
        if not git_dir.exists():
            import subprocess
            subprocess.run(["git", "init"], cwd=str(cwd), capture_output=True)

        # Codex CLI: exec for non-interactive, --full-auto for auto-approve,
        # --json for JSONL output
        full_prompt = f"{self._system_prompt}\n\n{prompt}"

        cmd = [
            *bin_path.split(),
            "exec",
            "--full-auto",
            "--json",
            "-C", str(cwd),
            full_prompt,
        ]
        if self._model:
            cmd.extend(["-m", self._model])

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse Codex JSONL output
        events = self._parse_jsonl(stdout)
        if events:
            result.response, result.tool_calls = self._parse_codex_events(events)
        else:
            # Fall back to plain text
            result.response = stdout

        return result

    @staticmethod
    def _parse_codex_events(events: list[dict]) -> tuple[str, list[dict]]:
        """Parse Codex CLI JSONL events into response + tool calls."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []

        for event in events:
            event_type = event.get("type", "")

            # Codex JSONL event types
            if event_type == "message":
                role = event.get("role", "")
                if role == "assistant":
                    content = event.get("content", "")
                    if isinstance(content, str) and content:
                        response_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                response_parts.append(part.get("text", ""))
            elif event_type in ("function_call", "tool_call"):
                tool_calls.append({
                    "tool": event.get("name", event.get("function", {}).get("name", "")),
                    "input": str(event.get("arguments", event.get("input", "")))[:500],
                })
            elif event_type == "exec":
                tool_calls.append({
                    "tool": "shell",
                    "input": str(event.get("command", event.get("args", "")))[:500],
                })

            # Handle nested content blocks (OpenAI format)
            if "output" in event and isinstance(event["output"], list):
                for item in event["output"]:
                    if isinstance(item, dict):
                        if item.get("type") == "message":
                            text = item.get("content", "")
                            if isinstance(text, str) and text:
                                response_parts.append(text)

        return "\n".join(response_parts), tool_calls
