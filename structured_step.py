"""Structured step schema based on Fara-7B and Agentic Organization research."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuredStep:
    """
    Structured representation of a test step.
    
    Based on Fara-7B research:
    - Connect "thoughts" to concrete interactions
    - Include expected visual signals
    - Track pre/post state for verification
    """
    step_id: int
    raw_text: str
    action_type: str  # navigate, click, type, verify, assert, scroll, wait
    
    # Target element
    target_selector: str | None = None  # CSS/XPath if known
    target_description: str | None = None  # Natural language description
    
    # Input data
    input_data: str | None = None  # Text to type, URL to open
    
    # Expected outcomes (for verification)
    expected_visual_signal: str | None = None  # What should appear on screen
    expected_text: str | None = None  # Text that should be visible
    expected_state_change: str | None = None  # e.g., "modal opens", "page navigates"
    
    # State tracking
    pre_state: dict[str, Any] | None = None  # Expected state before action
    post_state: dict[str, Any] | None = None  # Expected state after action
    
    # Timing
    timing_hint: tuple[float, float] | None = None  # Expected time window (start, end)
    
    # Dependencies
    depends_on: list[int] = field(default_factory=list)  # Step IDs this depends on
    
    # Metadata
    source: str | None = None  # Where step came from (log, gherkin, etc.)
    confidence: float = 1.0  # How confident are we in this parsing


class StepParser:
    """
    Parse raw step text into StructuredStep objects.
    
    Uses patterns to extract:
    - Action type
    - Target element
    - Expected outcomes
    - Input data
    """
    
    # Action type patterns
    ACTION_PATTERNS = {
        "navigate": [
            r"navigate(?:s)? to",
            r"go(?:es)? to",
            r"open(?:s)?.*url",
            r"visit(?:s)?",
        ],
        "click": [
            r"click(?:s)?(?:\s+on)?",
            r"press(?:es)?",
            r"tap(?:s)?",
            r"select(?:s)?(?!\s+size)",  # "select" but not "select size"
        ],
        "type": [
            r"type(?:s)?",
            r"enter(?:s)?(?:\s+text)?",
            r"input(?:s)?",
            r"fill(?:s)?(?:\s+in)?",
        ],
        "verify": [
            r"verify",
            r"confirm(?:s)?",
            r"check(?:s)?(?!\s+out)",  # "check" but not "checkout"
            r"ensure(?:s)?",
        ],
        "assert": [
            r"assert(?:s)?",
            r"should(?:\s+be)?(?:\s+visible)?",
            r"must(?:\s+be)?",
            r"expect(?:s)?",
        ],
        "scroll": [
            r"scroll(?:s)?",
            r"swipe(?:s)?",
        ],
        "wait": [
            r"wait(?:s)?",
            r"pause(?:s)?",
            r"delay(?:s)?",
        ],
        "submit": [
            r"submit(?:s)?",
            r"send(?:s)?",
        ],
        "close": [
            r"close(?:s)?",
            r"dismiss(?:es)?",
        ],
    }
    
    # Element patterns
    ELEMENT_PATTERNS = {
        "button": [r"button", r"btn"],
        "input": [r"input(?:\s+field)?", r"text(?:\s+field)?", r"search(?:\s+bar)?"],
        "link": [r"link", r"anchor"],
        "modal": [r"modal", r"popup", r"dialog", r"overlay"],
        "dropdown": [r"dropdown", r"select(?:\s+box)?", r"combobox"],
        "checkbox": [r"checkbox", r"check(?:\s+box)?"],
        "menu": [r"menu", r"navigation"],
        "icon": [r"icon", r"magnifier", r"search(?:\s+icon)?"],
    }
    
    def parse(self, raw_text: str, step_id: int = 1) -> StructuredStep:
        """Parse raw step text into a StructuredStep."""
        lowered = raw_text.lower()
        
        # Extract action type
        action_type = self._extract_action_type(lowered)
        
        # Extract target
        target_desc, target_selector = self._extract_target(raw_text)
        
        # Extract input data (quoted strings)
        input_data = self._extract_input_data(raw_text)
        
        # Extract expected signals
        expected_signal = self._extract_expected_signal(raw_text)
        expected_text = self._extract_expected_text(raw_text)
        
        return StructuredStep(
            step_id=step_id,
            raw_text=raw_text,
            action_type=action_type,
            target_description=target_desc,
            target_selector=target_selector,
            input_data=input_data,
            expected_visual_signal=expected_signal,
            expected_text=expected_text,
        )
    
    def _extract_action_type(self, text: str) -> str:
        """Extract the primary action type from step text."""
        for action, patterns in self.ACTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return action
        return "generic"
    
    def _extract_target(self, text: str) -> tuple[str | None, str | None]:
        """Extract target element description and selector."""
        target_desc = None
        target_selector = None
        
        # Look for quoted targets (e.g., click on "Login")
        quoted_match = re.search(r'"([^"]+)"', text)
        if quoted_match:
            target_desc = quoted_match.group(1)
        
        # Look for element type patterns
        for elem_type, patterns in self.ELEMENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(rf"the\s+{pattern}", text, re.IGNORECASE)
                if match:
                    if target_desc:
                        target_desc = f"{elem_type}: {target_desc}"
                    else:
                        target_desc = elem_type
                    break
        
        # Look for CSS selectors (rare in natural language but possible)
        selector_match = re.search(r'\[.*?\]|#[\w-]+|\.[\w-]+', text)
        if selector_match:
            target_selector = selector_match.group(0)
        
        return target_desc, target_selector
    
    def _extract_input_data(self, text: str) -> str | None:
        """Extract input data (typically quoted strings)."""
        # Look for quoted strings after action keywords
        patterns = [
            r'(?:type|enter|input|fill)\s+(?:text\s+)?["\']([^"\']+)["\']',
            r'["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_expected_signal(self, text: str) -> str | None:
        """Extract expected visual signal from step."""
        # Look for "should see", "should display", etc.
        patterns = [
            r'should\s+(?:see|display|show)\s+(.+?)(?:\.|$)',
            r'(?:verify|confirm|check)\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'(.+?)\s+should\s+be\s+visible',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_expected_text(self, text: str) -> str | None:
        """Extract expected text content."""
        # Look for text in quotes after "containing", "with", "including"
        patterns = [
            r'(?:containing|with|including)\s+["\']([^"\']+)["\']',
            r'text\s+["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


def parse_steps(raw_steps: list[str]) -> list[StructuredStep]:
    """Parse multiple raw steps into StructuredStep objects."""
    parser = StepParser()
    return [parser.parse(step, idx + 1) for idx, step in enumerate(raw_steps)]


def build_dependency_graph(steps: list[StructuredStep]) -> dict[int, list[int]]:
    """
    Build a dependency graph between steps.
    
    Heuristics:
    - Action steps depend on previous navigation/verification
    - Submit depends on previous type/fill
    - Assert depends on previous action
    """
    dependencies: dict[int, list[int]] = {}
    
    for i, step in enumerate(steps):
        deps = []
        
        # Actions generally depend on previous step
        if i > 0:
            deps.append(steps[i - 1].step_id)
        
        # Submit depends on type actions
        if step.action_type == "submit":
            for j in range(i - 1, -1, -1):
                if steps[j].action_type == "type":
                    deps.append(steps[j].step_id)
                    break
        
        # Assert depends on action being verified
        if step.action_type == "assert":
            for j in range(i - 1, -1, -1):
                if steps[j].action_type in ["click", "type", "submit"]:
                    deps.append(steps[j].step_id)
                    break
        
        dependencies[step.step_id] = list(set(deps))
        step.depends_on = dependencies[step.step_id]
    
    return dependencies
