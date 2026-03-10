"""Gemini CLI agent backend — runs tasks via the Google Gemini CLI."""

from __future__ import annotations

import logging
from pathlib import Path

from src.eval.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class GeminiCLIAgent(BaseAgent):
    """Runs tasks through Gemini CLI (npx @google/gemini-cli)."""

    def name(self) -> str:
        return "gemini-cli"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        bin_path = self._find_binary("@google/gemini-cli", "gemini")

        # Gemini CLI: -p for non-interactive, --yolo to auto-approve,
        # -o json for structured output
        full_prompt = f"{self._system_prompt}\n\n{prompt}"

        cmd = [
            *bin_path.split(),
            "-p", full_prompt,
            "--yolo",
            "-o", "json",
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        import os
        # Gemini CLI uses GEMINI_API_KEY or GOOGLE_API_KEY
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = env_key

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse Gemini JSON output (JSONL stream)
        events = self._parse_jsonl(stdout)
        if events:
            result.response, result.tool_calls = self._parse_gemini_events(events)
        else:
            # Fall back to plain text
            result.response = stdout

        return result

    @staticmethod
    def _parse_gemini_events(events: list[dict]) -> tuple[str, list[dict]]:
        """Parse Gemini CLI JSON events into response + tool calls."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []

        for event in events:
            event_type = event.get("type", "")

            # Gemini stream-json format
            if event_type == "text":
                response_parts.append(event.get("content", ""))
            elif event_type == "toolCall":
                tool_calls.append({
                    "tool": event.get("name", event.get("toolName", "")),
                    "input": str(event.get("args", event.get("input", "")))[:500],
                })
            elif event_type == "result" or event_type == "response":
                content = event.get("content", event.get("text", ""))
                if content:
                    response_parts.append(content)

            # Also handle nested message format
            if "content" in event and isinstance(event["content"], list):
                for part in event["content"]:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            response_parts.append(part.get("text", ""))
                        elif part.get("type") in ("toolCall", "tool_use"):
                            tool_calls.append({
                                "tool": part.get("name", ""),
                                "input": str(part.get("args", part.get("input", "")))[:500],
                            })

        return "\n".join(response_parts), tool_calls
