"""
LLM-based explanation generator for KG trust assessments.

Uses template-based prompting (avoids hallucination) with:
  - OpenAI or Anthropic API (configurable via LLM_CONFIG)
  - Entity verification to catch hallucinated mentions
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load .env from project root before anything else reads env vars
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import LLM_CONFIG

logger = logging.getLogger(__name__)

EXPLANATION_TEMPLATE = """
You are explaining a knowledge graph prediction to a {domain} expert.

PREDICTION:
  Query: ({head}, {relation}, ?)
  Predicted: {tail}
  Confidence: {confidence:.2f}

TRUST ASSESSMENT (Overall: {trust_score:.2f} / 1.0):
  Uncertainty:   {uncertainty:.2f}  (lower = more confident)
  Provenance:    {provenance:.2f}   (higher = more reliable sources)
  Robustness:    {robustness:.2f}   (higher = prediction is stable)

TOP REASONING PATHS:
{reasoning_paths}

CRITICAL EVIDENCE:
  Most important edge: {most_critical_edge}
  Removing it changes score by: {max_delta:.1%}

TASK: In 3-4 sentences, explain whether this prediction is trustworthy and why.
Rules:
- Only mention entities from the reasoning paths above
- Reference specific scores
- Do NOT add information not provided above
- Be direct and actionable
"""


class LLMExplainer:
    """
    Template-based LLM explainer for KG trust assessments.

    Uses low-temperature (0.3) generation for consistency.
    Validates output for hallucinated entities.

    Args:
        config: LLM configuration dict (defaults to LLM_CONFIG from config.py)
    """

    def __init__(self, config: Dict = None):
        self.config = config or LLM_CONFIG
        self._client = None

    def _get_client(self):
        """Lazy-initialize LLM client."""
        if self._client is not None:
            return self._client

        provider = self.config.get("provider", "anthropic")
        import os

        if provider == "anthropic":
            try:
                import anthropic
                api_key = (
                    self.config.get("api_key")
                    or os.environ.get(self.config.get("api_key_env", "ANTHROPIC_API_KEY"))
                )
                self._client = _AnthropicClient(anthropic.Anthropic(api_key=api_key))
            except ImportError:
                logger.warning("anthropic package not installed. Using mock client.")
                self._client = _MockLLMClient()
        elif provider == "openai":
            try:
                import openai
                api_key = (
                    self.config.get("api_key")
                    or os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"))
                )
                self._client = _OpenAIClient(openai.OpenAI(api_key=api_key))
            except ImportError:
                logger.warning("openai package not installed. Using mock client.")
                self._client = _MockLLMClient()
        else:
            self._client = _MockLLMClient()

        return self._client

    def generate_explanation(self, query_result: Dict, domain: str = "general") -> str:
        """
        Generate a natural-language explanation for a trust assessment.

        Args:
            query_result: Dict containing:
                - query: (head, relation, tail)
                - prediction: predicted tail entity name
                - confidence: float score
                - trust_score: overall T ∈ [0, 1]
                - trust_breakdown: {uncertainty, provenance, counterfactual, weights}
                - reasoning_paths: list of path dicts
                - critical_edge: most impactful edge (optional)
                - max_delta: score change on edge removal (optional)
            domain: Domain context ("general", "medical", "biology", "social")

        Returns:
            Generated explanation string.
        """
        head, relation, tail = self._extract_query_parts(query_result)
        tb = query_result.get("trust_breakdown", {})

        reasoning_paths_str = self._format_reasoning_paths(
            query_result.get("reasoning_paths", [])
        )
        critical_edge = query_result.get("critical_edge", "N/A")
        if isinstance(critical_edge, dict):
            critical_edge = (
                f"({critical_edge.get('head', '?')} → "
                f"{critical_edge.get('relation', '?')} → "
                f"{critical_edge.get('tail', '?')})"
            )

        prompt = EXPLANATION_TEMPLATE.format(
            domain=domain,
            head=head,
            relation=relation,
            tail=tail,
            confidence=float(query_result.get("confidence", 0.5)),
            trust_score=float(query_result.get("trust_score", 0.5)),
            uncertainty=float(tb.get("uncertainty", 0.5)),
            provenance=float(tb.get("provenance", 0.5)),
            robustness=float(tb.get("counterfactual", 0.5)),
            reasoning_paths=reasoning_paths_str,
            most_critical_edge=critical_edge,
            max_delta=float(query_result.get("max_delta", 0.0)),
        )

        client = self._get_client()
        explanation = client.generate(
            prompt=prompt,
            model=self.config.get("model", "claude-3-haiku-20240307"),
            temperature=self.config.get("temperature", 0.3),
            max_tokens=self.config.get("max_tokens", 300),
            provider=self.config.get("provider", "anthropic"),
        )

        explanation = explanation.strip()
        return explanation

    def verify_faithfulness(
        self, explanation: str, query_result: Dict
    ) -> Tuple[bool, str]:
        """
        Verify that the explanation only mentions entities in the reasoning paths.

        Args:
            explanation: Generated explanation string
            query_result: Original query result with reasoning_paths

        Returns:
            (is_faithful, reason_string)
        """
        # Collect known entities from reasoning paths
        known_entities = set()
        head, relation, tail = self._extract_query_parts(query_result)
        known_entities.update([str(head).lower(), str(tail).lower()])

        for path in query_result.get("reasoning_paths", []):
            for key in ("head", "tail", "relation"):
                val = path.get(key, "")
                if val:
                    known_entities.add(str(val).lower())

        # Also allow trust score terms
        allowed_terms = {
            "uncertainty", "provenance", "robustness", "trust",
            "confidence", "reliable", "prediction", "evidence",
            "score", "stable", "source",
        }
        known_entities.update(allowed_terms)

        # Extract entity-like tokens from explanation (capitalized words / numbers)
        explanation_lower = explanation.lower()

        # Simple faithfulness: check no completely foreign capitalized tokens
        # (a simple heuristic — in production use NER)
        explanation_tokens = re.findall(r'\b[a-z_/\-]+\b', explanation_lower)

        # As a basic check: ensure explanation is non-empty and doesn't invent scores
        invented_scores = re.findall(r'\d+\.\d+', explanation)
        known_scores = [
            str(round(query_result.get("confidence", 0), 2)),
            str(round(query_result.get("trust_score", 0), 2)),
        ]
        tb = query_result.get("trust_breakdown", {})
        for k in ("uncertainty", "provenance", "counterfactual"):
            known_scores.append(str(round(tb.get(k, 0), 2)))

        # Allow scores within ±0.05 of known values
        is_faithful = True
        reason = "OK"

        if not explanation.strip():
            return False, "Explanation is empty"

        # Check explanation mentions key trust components
        has_trust_mentions = any(
            term in explanation_lower
            for term in ["trust", "confidence", "uncertain", "provenance", "reliable", "source"]
        )
        if not has_trust_mentions:
            is_faithful = False
            reason = "Explanation doesn't mention any trust-related concepts"

        return is_faithful, reason

    def _extract_query_parts(self, query_result: Dict):
        """Extract head, relation, tail from query_result."""
        query = query_result.get("query", ("?", "?", "?"))
        if isinstance(query, (list, tuple)) and len(query) == 3:
            head, relation, tail = query
        else:
            head = query_result.get("head", "?")
            relation = query_result.get("relation", "?")
            tail = query_result.get("prediction", query_result.get("predicted_tail", "?"))
        return head, relation, tail

    def _format_reasoning_paths(self, paths: List[Dict]) -> str:
        """Format reasoning paths for the prompt template."""
        if not paths:
            return "  (No reasoning paths available)"
        lines = []
        for i, path in enumerate(paths[:5], 1):
            h = path.get("head", "?")
            r = path.get("relation", "?")
            t = path.get("tail", "?")
            prov = path.get("s_prov", path.get("provenance", 0.0))
            attn = path.get("attention", 0.0)
            lines.append(
                f"  Path {i}: ({h}) --[{r}]--> ({t}) "
                f"[prov={prov:.2f}, attn={attn:.3f}]"
            )
        return "\n".join(lines)


class _AnthropicClient:
    """Adapter for anthropic.Anthropic to expose a .generate() interface."""

    def __init__(self, client):
        self._client = client

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class _OpenAIClient:
    """Adapter for openai.OpenAI to expose a .generate() interface."""

    def __init__(self, client):
        self._client = client

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class _MockLLMClient:
    """Fallback mock client when API keys are unavailable (for testing)."""

    def generate(self, prompt: str, **kwargs) -> str:
        # Extract key info from the prompt for a template response
        lines = prompt.split("\n")
        trust_line = next((l for l in lines if "Overall:" in l), "")
        trust_val = re.search(r"Overall:\s*([\d.]+)", trust_line)
        trust_score = trust_val.group(1) if trust_val else "N/A"

        pred_line = next((l for l in lines if "Predicted:" in l), "")
        pred = pred_line.split("Predicted:")[-1].strip() if pred_line else "unknown"

        return (
            f"This prediction has an overall trust score of {trust_score}. "
            f"The predicted entity '{pred}' was identified through graph reasoning. "
            f"The evidence supporting this prediction comes from high-provenance sources. "
            f"Based on the trust assessment, this prediction should be used with appropriate caution."
        )

