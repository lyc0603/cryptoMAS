"""
Base agent class wrapping the OpenAI chat completion API.

All three agents (Crypto, News, Trading) inherit from BaseAgent.
Capability variants (zero-shot, CoT, skill-augmented, etc.) are
selected by passing a `capability` argument at construction time.

Memory system
-------------
Each agent maintains a rolling list of its own past outputs.  Before
every new call the most recent `memory_window` entries are prepended to
the user message so the agent can reason about trends and consistency
across weeks.  Memory can be serialised / restored for experiment
resumption.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

CAPABILITIES = {
    "zero_shot",
    "chain_of_thought",
    "skill_augmented",
}


class BaseAgent(ABC):
    """
    Abstract base for all MAS agents.

    Args:
        capability:     One of 'zero_shot', 'chain_of_thought', 'skill_augmented'.
        model:          OpenAI model ID (default: gpt-4o).
        temperature:    Sampling temperature.
        memory_window:  Number of past weekly outputs to include in each prompt.
                        Set to 0 to disable memory.
        api_key:        OpenAI API key. Falls back to OPENAI_API_KEY env var.
        max_retries:    Number of automatic retries on rate-limit / server errors
                        (passed directly to the OpenAI client; default 3).
                        The SDK uses exponential backoff between retries.
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        if capability not in CAPABILITIES:
            raise ValueError(f"capability must be one of {CAPABILITIES}, got '{capability}'")
        self.capability = capability
        self.model = model
        self.temperature = temperature
        self.memory_window = memory_window
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=max_retries,
        )

        # Memory: list of {"week": str, "output": Any}
        self.memory: list[dict] = []

    # ------------------------------------------------------------------
    # Subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Static system prompt that defines the agent's role."""

    @abstractmethod
    def build_user_message(self, context: dict) -> str:
        """Build the user-turn message from the provided context dict."""

    @abstractmethod
    def parse_response(self, content: str) -> Any:
        """
        Parse the model's text response into a structured output.
        Must RAISE on failure — exceptions propagate to the caller and
        terminate the experiment so no wrong result is ever saved.
        """

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _format_memory(self) -> str:
        """
        Render the last `memory_window` entries as a text block to
        prepend to the user message.
        """
        recent = self.memory[-self.memory_window:]
        lines = [
            "## Your past outputs (most recent first — use these to reason about trends):",
        ]
        for entry in reversed(recent):
            lines.append(f"\nWeek {entry['week']}:")
            lines.append(json.dumps(entry["output"], indent=2))
        return "\n".join(lines)

    def _store_memory(self, week: str, output: Any) -> None:
        self.memory.append({"week": week, "output": output})

    def get_memory_state(self) -> list[dict]:
        """Return a serialisable copy of the memory (for persistence)."""
        return list(self.memory)

    def set_memory_state(self, state: list[dict]) -> None:
        """Restore memory from a previously saved state."""
        self.memory = list(state)

    # ------------------------------------------------------------------
    # Capability-specific prompt decoration
    # ------------------------------------------------------------------

    def _decorate_system_prompt(self) -> str:
        base = self.system_prompt
        if self.capability == "chain_of_thought":
            base += (
                "\n\nReasoning instruction: Before giving your final answer, "
                "think step-by-step inside <reasoning>...</reasoning> tags. "
                "Your final structured output must appear after the closing </reasoning> tag."
            )
        elif self.capability == "skill_augmented":
            base += (
                "\n\nYou have access to the following analytical tools which you may "
                "reference in your reasoning:\n"
                + self._skill_descriptions()
            )
        return base

    def _skill_descriptions(self) -> str:
        """Override in subclass to list domain-specific skill descriptions."""
        return ""

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def run(self, context: dict, memorize: bool = True) -> Any:
        """
        Execute the agent for a given context.

        Memory from previous calls is automatically prepended to the
        user message.  The new output is appended to memory only when
        `memorize=True` (default).  Pass `memorize=False` for
        intermediate refinement rounds where only the final output
        should be remembered.

        Args:
            context:  Dict of inputs — content depends on agent type.
                      Must include a "week" key for memory book-keeping.
            memorize: Whether to store this call's output in memory.

        Returns:
            Parsed structured output from parse_response().
        """
        system = self._decorate_system_prompt()
        user_msg = self.build_user_message(context)

        # Prepend memory block when available
        if self.memory and self.memory_window > 0:
            user_msg = self._format_memory() + "\n\n---\n\n" + user_msg

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        logger.debug(
            "%s.run | week=%s model=%s capability=%s memory_entries=%d memorize=%s",
            self.__class__.__name__,
            context.get("week", "?"),
            self.model,
            self.capability,
            len(self.memory),
            memorize,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content
        logger.debug("Raw response:\n%s", content)

        # Raises on failure — caller is responsible for terminating the run.
        result = self.parse_response(content)

        week = context.get("week")
        if memorize and week:
            self._store_memory(week, result)

        return result

    # ------------------------------------------------------------------
    # Helper: extract JSON from model output
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Extract the first JSON object or array from a text string.
        Strips markdown code fences if present.

        Chain-of-thought models should place JSON *after* </reasoning>, but
        sometimes emit it *inside* the reasoning block.  We handle both:
          1. Strip everything up to </reasoning>, try to parse what follows.
          2. If nothing remains, fall back to searching within the reasoning block.
        """
        reasoning_inner: str | None = None

        if "<reasoning>" in text and "</reasoning>" in text:
            start = text.index("<reasoning>") + len("<reasoning>")
            end   = text.index("</reasoning>")
            reasoning_inner = text[start:end]
            text = text[end + len("</reasoning>"):]

        def _parse_fenced(src: str) -> Any:
            """Strip optional markdown fences and parse JSON."""
            for fence in ("```json", "```"):
                if fence in src:
                    src = src.split(fence, 1)[-1]
                    src = src.rsplit("```", 1)[0]
                    break
            return json.loads(src.strip())

        # Primary: text after </reasoning>
        if text.strip():
            return _parse_fenced(text)

        # Fallback: JSON was placed inside the reasoning block (model formatting error)
        if reasoning_inner:
            return _parse_fenced(reasoning_inner)

        raise ValueError("No JSON content found in model response")
