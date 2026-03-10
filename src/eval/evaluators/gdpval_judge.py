"""GDPval evaluator using Gemini Pro as a frozen judge.

Sends the task prompt, the agent's response, and the rubric to Gemini,
which scores each criterion individually. This replaces the keyword-based
heuristic with an LLM judge for accurate rubric evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig

from src.eval.evaluators.base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)

# Default model — frozen across the entire experiment
DEFAULT_JUDGE_MODEL = "gemini-2.5-pro-preview-06-05"

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial grading judge for professional work tasks.

You will receive:
1. TASK: The original task prompt given to an AI assistant.
2. RESPONSE: The assistant's full response (text output and tool usage).
3. RUBRIC: A list of scoring criteria in [+N] / [-N] format.

Your job is to evaluate the RESPONSE against each rubric criterion and assign points.

Rules:
- Score each criterion independently.
- Award full points ONLY if the criterion is clearly and fully met.
- Award 0 if the criterion is not met or only partially met.
- For negative criteria ([-N]), apply the penalty if the condition is true.
- Base your judgment on what the response actually demonstrates, not what it claims.
- If the task required file deliverables (Excel, Word, etc.) and the response only describes
  them in text without creating them, criteria about file format/structure are NOT met.
- Be strict: vague or partial matches do not earn points.

Output your evaluation as JSON with this exact structure:
{
  "criteria": [
    {
      "criterion": "[+2] The exact criterion text",
      "points_awarded": 2,
      "points_possible": 2,
      "reasoning": "Brief explanation of why points were awarded or not"
    }
  ],
  "total_score": 16,
  "max_score": 20,
  "overall_notes": "Brief overall assessment"
}

Return ONLY the JSON object, no other text."""


class GDPvalJudgeEvaluator(BaseEvaluator):
    """Scores GDPval tasks using Gemini Pro as a frozen judge.

    The judge reads the task prompt, the agent's response, and the full
    rubric, then scores each criterion with reasoning.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable, or pass api_key= to the constructor."
            )
        self._client = genai.Client(api_key=key)

    def name(self) -> str:
        return "gdpval_judge"

    def evaluate(
        self,
        task_id: str,
        prompt: str,
        response: str,
        reference: str,
        **kwargs: Any,
    ) -> EvalResult:
        """Send task + response + rubric to Gemini for scoring."""
        if not response.strip():
            return EvalResult(
                task_id=task_id,
                score=0.0,
                max_score=0.0,
                error="Empty response",
            )

        judge_prompt = self._build_judge_prompt(prompt, response, reference)

        for attempt in range(self._max_retries + 1):
            try:
                judge_response = self._call_gemini(judge_prompt)
                result = self._parse_judge_response(task_id, judge_response, response)
                return result
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Judge parse error on task %s (attempt %d): %s",
                    task_id, attempt + 1, e,
                )
                if attempt == self._max_retries:
                    return EvalResult(
                        task_id=task_id,
                        score=0.0,
                        max_score=0.0,
                        response_text=response[:2000],
                        error=f"Judge parse failed after {self._max_retries + 1} attempts: {e}",
                        metadata={"raw_judge_response": judge_response[:2000]},
                    )
            except Exception as e:
                logger.error("Gemini API error on task %s: %s", task_id, e)
                if attempt == self._max_retries:
                    return EvalResult(
                        task_id=task_id,
                        score=0.0,
                        max_score=0.0,
                        response_text=response[:2000],
                        error=f"Gemini API error: {e}",
                    )
                time.sleep(2 ** attempt)  # Exponential backoff

        # Should not reach here, but satisfy type checker
        return EvalResult(task_id=task_id, score=0.0, max_score=0.0, error="Unexpected")

    def _build_judge_prompt(self, task_prompt: str, response: str, rubric: str) -> str:
        return f"""## TASK
{task_prompt}

## RESPONSE
{response}

## RUBRIC
{rubric}

Evaluate the RESPONSE against each criterion in the RUBRIC. Return JSON only."""

    def _call_gemini(self, prompt: str) -> str:
        config = GenerateContentConfig(
            temperature=self._temperature,
            system_instruction=JUDGE_SYSTEM_PROMPT,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def _parse_judge_response(
        self, task_id: str, raw_response: str, agent_response: str
    ) -> EvalResult:
        """Parse the JSON output from the judge."""
        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        total_score = float(data["total_score"])
        max_score = float(data["max_score"])

        breakdown: dict[str, float] = {}
        criteria_details: list[dict[str, Any]] = []
        for c in data.get("criteria", []):
            criterion_text = c.get("criterion", "unknown")[:80]
            points = float(c.get("points_awarded", 0))
            breakdown[criterion_text] = points
            criteria_details.append({
                "criterion": c.get("criterion", ""),
                "points_awarded": points,
                "points_possible": float(c.get("points_possible", 0)),
                "reasoning": c.get("reasoning", ""),
            })

        return EvalResult(
            task_id=task_id,
            score=total_score,
            max_score=max_score,
            rubric_breakdown=breakdown,
            response_text=agent_response[:2000],
            metadata={
                "num_criteria": len(data.get("criteria", [])),
                "criteria_met": sum(
                    1 for c in data.get("criteria", [])
                    if float(c.get("points_awarded", 0)) > 0
                ),
                "overall_notes": data.get("overall_notes", ""),
                "criteria_details": criteria_details,
                "judge_model": self._model,
            },
        )
