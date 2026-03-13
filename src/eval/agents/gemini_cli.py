"""Gemini CLI agent backend — runs tasks via the Google Gemini CLI.

Gemini -o json format (single JSON object):
  {
    "session_id": "...",
    "response": "The file has been created.",
    "stats": {
      "tools": {"byName": {"write_file": {"count": 1, ...}}},
      "models": {"gemini-3-flash-preview": {"tokens": {...}}}
    }
  }

For richer traces, we also try -o stream-json which emits JSONL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class GeminiCLIAgent(BaseAgent):
    """Runs tasks through Gemini CLI (npx @google/gemini-cli)."""

    def name(self) -> str:
        return "gemini-cli"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        bin_path = self._find_binary("@google/gemini-cli", "gemini")

        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        # Gemini CLI: -p for non-interactive, --yolo to auto-approve,
        # -o stream-json for JSONL with tool call details
        # --sandbox false so it uses the real cwd
        full_prompt = f"{self._system_prompt}\n\nYour working directory is: {cwd}\nAll files MUST be saved there.\n\n{prompt}"

        cmd = [
            *bin_path.split(),
            "-p", full_prompt,
            "--yolo",
            "-o", "stream-json",
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Try parsing as JSONL (stream-json format) first
        events = self._parse_jsonl(stdout)
        if events:
            result.response, result.tool_calls, result.messages = self._parse_gemini_events(events)
            return result

        # Fall back to single JSON object (-o json format)
        try:
            data = json.loads(stdout)
            result.response = data.get("response", "")
            stats = data.get("stats", {})
            tools = stats.get("tools", {})
            for tool_name, info in tools.get("byName", {}).items():
                result.tool_calls.append({
                    "tool": tool_name,
                    "input": f"count={info.get('count', 0)} success={info.get('success', 0)}",
                })
                result.messages.append({
                    "role": "assistant", "type": "tool_use",
                    "tool": tool_name, "count": info.get("count", 0),
                })
        except json.JSONDecodeError:
            result.response = stdout

        return result

    @staticmethod
    def _parse_gemini_events(events: list[dict[str, Any]]) -> tuple[str, list[dict], list[dict]]:
        """Parse Gemini CLI stream-json JSONL into response + tool calls + messages."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []
        messages: list[dict] = []

        for event in events:
            event_type = event.get("type", "")

            if event_type == "message":
                role = event.get("role", "")
                content = event.get("content", "")
                if role == "assistant" and isinstance(content, str) and content:
                    response_parts.append(content)
                    messages.append({"role": "assistant", "type": "text", "content": content})
                elif role == "user" and isinstance(content, str) and content:
                    messages.append({"role": "user", "type": "text", "content": content[:2000]})

            elif event_type == "tool_use":
                tool_name = event.get("tool_name", "")
                params = event.get("parameters", {})
                tool_calls.append({"tool": tool_name, "input": str(params)[:500]})
                messages.append({
                    "role": "assistant", "type": "tool_use",
                    "tool": tool_name, "input": str(params)[:1000],
                })

            elif event_type == "tool_result":
                output = event.get("output", "")
                status = event.get("status", "")
                tool_calls.append({"tool": "_result", "input": output[:500], "status": status})
                messages.append({
                    "role": "tool", "type": "tool_result",
                    "content": output[:2000], "status": status,
                })

            elif event_type == "result":
                stats = event.get("stats", {})
                if stats:
                    tool_calls.append({
                        "tool": "_usage",
                        "input": f"in={stats.get('input_tokens', 0)} out={stats.get('output_tokens', 0)} cached={stats.get('cached', 0)} tool_calls={stats.get('tool_calls', 0)}",
                    })
                    messages.append({
                        "role": "system", "type": "usage",
                        "input_tokens": stats.get("input_tokens", 0),
                        "output_tokens": stats.get("output_tokens", 0),
                        "duration_ms": stats.get("duration_ms", 0),
                    })

        return "\n".join(response_parts), tool_calls, messages
