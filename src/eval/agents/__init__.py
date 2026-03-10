"""Agent backends for GDPval evaluation."""

from src.eval.agents.base import AgentResult, BaseAgent
from src.eval.agents.claude_code import ClaudeCodeAgent
from src.eval.agents.codex import CodexAgent
from src.eval.agents.gemini_cli import GeminiCLIAgent

AGENTS: dict[str, type[BaseAgent]] = {
    "claude": ClaudeCodeAgent,
    "gemini": GeminiCLIAgent,
    "codex": CodexAgent,
}

__all__ = [
    "AGENTS",
    "AgentResult",
    "BaseAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "GeminiCLIAgent",
]
