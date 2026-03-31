"""
Base agent class wrapping the OpenAI and Anthropic chat APIs.

All three agents (Crypto, News, Trading) inherit from BaseAgent.
Capability variants (zero-shot, CoT, skill-augmented, etc.) are
selected by passing a `capability` argument at construction time.

Model routing
-------------
- OpenAI Chat Completions API: gpt-4o and other gpt-* models
- OpenAI Responses API:        gpt-5 (does not accept temperature)
- Anthropic Messages API:      claude-* models

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

import anthropic as _anthropic
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------------
# Shared JSON schemas used by agents that produce the same output shape.
# -----------------------------------------------------------------------

# One signal object per asset (CryptoAgent family)
_SIGNAL_ITEM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "symbol":     {"type": "string"},
        "signal":     {"type": "number"},
        "confidence": {"type": "number"},
        "rationale":  {"type": "string"},
    },
    "required": ["symbol", "signal", "confidence", "rationale"],
    "additionalProperties": False,
}

CRYPTO_OUTPUT_SCHEMA: dict = {
    "type": "array",
    "items": _SIGNAL_ITEM_SCHEMA,
}

# One action object per asset (TradingAgent / SingleAgent)
_ACTION_ITEM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "symbol":    {"type": "string"},
        "action":    {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["symbol", "action", "rationale"],
    "additionalProperties": False,
}

TRADING_OUTPUT_SCHEMA: dict = {
    "type": "array",
    "items": _ACTION_ITEM_SCHEMA,
}

# NewsAgent output (object root — no wrapping needed)
NEWS_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "week":              {"type": "string"},
        "overall_sentiment": {"type": "number"},
        "overall_rationale": {"type": "string"},
        "coin_signals": {
            "type": "array",
            "items": _SIGNAL_ITEM_SCHEMA,
        },
    },
    "required": ["week", "overall_sentiment", "overall_rationale", "coin_signals"],
    "additionalProperties": False,
}

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

        # Instantiate the appropriate client based on model family
        if model.startswith("claude-"):
            self._provider = "anthropic"
            self._anthropic_client = _anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                max_retries=max_retries,
            )
        else:
            self._provider = "openai"
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

    @property
    @abstractmethod
    def output_schema(self) -> dict:
        """
        JSON Schema for this agent's output.  Used to enforce structured
        output at the API level (preventing invalid JSON from the model).
        Array schemas are automatically wrapped in an object envelope when
        required by the API.
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

        # Route to the correct provider and API endpoint.
        # All paths enforce the agent's JSON schema to prevent invalid JSON.
        _RESPONSES_API_MODELS = {"gpt-5"}
        schema = self.output_schema
        api_schema, wrapped = self._wrap_schema(schema)

        if self._provider == "anthropic":
            # Anthropic: use tool calling to enforce the output schema.
            tool = {
                "name": "report_output",
                "description": "Report the structured analysis result.",
                "input_schema": api_schema,
            }
            response = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=messages[0]["content"],
                messages=[{"role": "user", "content": messages[1]["content"]}],
                temperature=self.temperature,
                tools=[tool],
                tool_choice={"type": "tool", "name": "report_output"},
            )
            tool_block = next(b for b in response.content if b.type == "tool_use")
            raw = tool_block.input
            content = json.dumps(raw["items"] if wrapped else raw)
        elif self.model in _RESPONSES_API_MODELS:
            response = self.client.responses.create(
                model=self.model,
                instructions=messages[0]["content"],
                input=messages[1]["content"],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "schema": api_schema,
                        "strict": True,
                    }
                },
            )
            raw_text = response.output_text
            content = json.dumps(json.loads(raw_text)["items"]) if wrapped else raw_text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": api_schema,
                        "strict": True,
                    },
                },
            )
            raw_text = response.choices[0].message.content
            content = json.dumps(json.loads(raw_text)["items"]) if wrapped else raw_text
        logger.debug("Raw response:\n%s", content)

        # Raises on failure — caller is responsible for terminating the run.
        result = self.parse_response(content)

        week = context.get("week")
        if memorize and week:
            self._store_memory(week, result)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_schema(schema: dict) -> tuple[dict, bool]:
        """
        OpenAI strict JSON schema and Anthropic tool calling both require an
        object at the root level.  If the agent's schema is an array, wrap it
        as {"type":"object","properties":{"items":<schema>},...} and signal
        the caller to unwrap the response after decoding.

        Returns (api_schema, was_wrapped).
        """
        if schema.get("type") == "array":
            return (
                {
                    "type": "object",
                    "properties": {"items": schema},
                    "required": ["items"],
                    "additionalProperties": False,
                },
                True,
            )
        return schema, False

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
            src = src.strip()
            # Use raw_decode so trailing prose / extra JSON blocks are ignored
            obj, _ = json.JSONDecoder().raw_decode(src)
            return obj

        # Primary: text after </reasoning>
        if text.strip():
            return _parse_fenced(text)

        # Fallback: JSON was placed inside the reasoning block (model formatting error)
        if reasoning_inner:
            return _parse_fenced(reasoning_inner)

        raise ValueError("No JSON content found in model response")
