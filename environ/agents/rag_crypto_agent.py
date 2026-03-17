"""
RAG-augmented Crypto Agent.

Extends CryptoAgent: when "rag_examples" is present in the context dict,
historical market analogues are injected into the prompt before the final
instruction, giving the agent empirical reference points for calibrating
its per-asset signals.

If "rag_examples" is absent the agent behaves identically to CryptoAgent.
"""

from .crypto_agent import CryptoAgent, _SYSTEM

_RAG_ADDENDUM = (
    "\n\nYou will also receive historical analogues for each asset — past weeks "
    "with a similar price, volume, and market-cap profile — along with their "
    "realised next-week returns. Use these as additional evidence when assessing "
    "the direction and strength of each asset's signal."
)

_FINAL_INSTR = "Analyse each asset and return the JSON signal array."


class RAGCryptoAgent(CryptoAgent):
    """CryptoAgent augmented with retrieved historical market analogues."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM + _RAG_ADDENDUM

    def build_user_message(self, context: dict) -> str:
        base         = super().build_user_message(context)
        rag_examples = context.get("rag_examples", {})

        if not rag_examples:
            return base

        lines = ["\nHistorical analogues by market similarity:\n"]
        for sym in sorted(rag_examples):
            text = rag_examples[sym]
            if text:
                lines.append(f"[{sym}]\n{text}\n")

        rag_block = "\n".join(lines)
        return base.replace(_FINAL_INSTR, rag_block + "\n" + _FINAL_INSTR)
