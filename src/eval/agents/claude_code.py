"""Claude Code agent backend — runs tasks via the Claude Code CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.eval.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAgent):
    """Runs tasks through Claude Code CLI (npx @anthropic-ai/claude-code)."""

    def name(self) -> str:
        return "claude-code"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        import os
        # Allow launching Claude Code from within another Claude Code session
        os.environ.pop("CLAUDECODE", None)

        bin_path = self._find_binary("@anthropic-ai/claude-code", "claude")

        cmd = [
            *bin_path.split(),
            "-p",
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--system-prompt", self._system_prompt,
            "--max-turns", str(self._max_turns),
            prompt,
        ]
        if self._model:
            cmd.insert(cmd.index("-p"), "--model")
            cmd.insert(cmd.index("-p"), self._model)

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse JSON output
        try:
            data = json.loads(stdout)
            # Claude Code JSON output has a "result" field with the final text
            if isinstance(data, dict):
                result.response = data.get("result", "")
                # Extract tool calls if present
                for msg in data.get("messages", []):
                    if msg.get("type") == "tool_use":
                        result.tool_calls.append({
                            "tool": msg.get("name", ""),
                            "input": str(msg.get("input", ""))[:500],
                        })
            elif isinstance(data, list):
                # Stream JSON format - list of events
                result.response, result.tool_calls = self._parse_claude_events(data)
        except json.JSONDecodeError:
            # Plain text output
            result.response = stdout

        return result

    @staticmethod
    def _parse_claude_events(events: list[dict]) -> tuple[str, list[dict]]:
        """Parse Claude Code stream-json events into response + tool calls."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []

        for event in events:
            msg_type = event.get("type", "")
            if msg_type == "assistant":
                for block in event.get("content", []):
                    if block.get("type") == "text":
                        response_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "tool": block.get("name", ""),
                            "input": str(block.get("input", ""))[:500],
                        })
            elif msg_type == "result":
                if event.get("result"):
                    response_parts.append(event["result"])

        return "\n".join(response_parts), tool_calls
