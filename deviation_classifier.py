"""Deviation classifier with 6-class taxonomy based on HALoGEN, BrowseComp, and HLE research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from analysis_agent import EvidenceStore, PlanStep, StepResult


class StepStatus:
    """
    6-class deviation taxonomy inspired by BrowseComp and HALoGEN benchmarks.
    
    - OBSERVED: Clear visual evidence and log support found.
    - PARTIALLY_OBSERVED: Only some expected signals or partial log matches found.
    - DEVIATION_SKIPPED: No evidence found, likely skipped due to prior failure.
    - DEVIATION_ALTERED: Evidence found but differs from expected plan.
    - HALLUCINATED: Logs claim success but no visual evidence exists.
    - UNCLEAR: Insufficient or conflicting evidence.
    """
    OBSERVED = "Observed"
    DEVIATION_SKIPPED = "Deviation-Skipped"
    DEVIATION_ALTERED = "Deviation-Altered"
    HALLUCINATED = "Hallucinated"
    UNCLEAR = "Unclear"
    PARTIALLY_OBSERVED = "Partially-Observed"


@dataclass
class ClassificationResult:
    """Result of deviation classification with confidence scoring."""
    status: str
    confidence: float  # 0.0 - 1.0
    evidence_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    expected_signal: str | None = None
    actual_signal: str | None = None
    timestamp_window: tuple[float, float] | None = None


class DeviationClassifier:
    """
    Centralized classifier for step deviations.
    
    Classification flow (HALoGEN-inspired):
    1. Check if previous step failed → Deviation-Skipped (cascade failure)
    2. Search for expected visual signal in evidence → Observed
    3. Check for log claims without visual evidence → Hallucinated
    4. Check for partial matches → Partially-Observed or Deviation-Altered
    5. Weak/no match → Unclear
    """

    # Keywords that indicate action completion in logs
    ACTION_KEYWORDS = {
        "navigate": ["navigated", "opened", "loaded", "page loaded"],
        "click": ["clicked", "pressed", "tapped", "selected"],
        "type": ["typed", "entered", "input", "filled", "text entered"],
        "submit": ["submitted", "sent", "completed"],
        "verify": ["verified", "confirmed", "checked", "asserted"],
        "scroll": ["scrolled", "scrolled down", "scrolled up"],
        "wait": ["waited", "paused", "delayed"],
    }

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.6
    LOW_CONFIDENCE = 0.3

    def classify(
        self,
        step: "PlanStep",
        evidence: "EvidenceStore",
        previous_result: Any = None,
        visual_matches: list | None = None,
        log_has_claim: bool = False,
        log_confirmed_success: bool = False,
    ) -> ClassificationResult:
        """
        Classify a step based on available evidence.

        Args:
            step: The plan step to evaluate
            evidence: All available evidence (screenshots, video frames)
            previous_result: Result of the previous step (for cascade detection)
            visual_matches: Evidence items that match expected signals
            log_has_claim: Whether logs claim this step was executed
            log_confirmed_success: Whether logs explicitly confirm SUCCESS

        Returns:
            ClassificationResult with status, confidence, and evidence
        """
        visual_matches = visual_matches or []

        # ─────────────────────────────────────────────────────────────────────
        # 1. Cascade failure detection (Deviation-Skipped)
        # ─────────────────────────────────────────────────────────────────────
        if previous_result and hasattr(previous_result, 'status'):
            if previous_result.status in [
                StepStatus.DEVIATION_SKIPPED,
                StepStatus.HALLUCINATED,
                StepStatus.DEVIATION_ALTERED,
            ]:
                # If no evidence for this step either, it was likely skipped
                if not visual_matches and not log_has_claim:
                    return ClassificationResult(
                        status=StepStatus.DEVIATION_SKIPPED,
                        confidence=0.9,
                        notes=[f"Previous step failed; no evidence for this step."],
                    )

        # ─────────────────────────────────────────────────────────────────────
        # 2. Strong visual evidence → Observed
        # ─────────────────────────────────────────────────────────────────────
        if visual_matches:
            confidence = self.calculate_confidence(
                visual_matches=len(visual_matches),
                log_support=log_has_claim,
                previous_step_passed=(previous_result is None or 
                                       getattr(previous_result, 'status', '') == StepStatus.OBSERVED),
                action_type=step.action if hasattr(step, 'action') else "",
            )
            return ClassificationResult(
                status=StepStatus.OBSERVED,
                confidence=confidence,
                evidence_paths=[str(getattr(m, 'path', m)) for m in visual_matches[:3]],
                notes=[f"Visual evidence found in {len(visual_matches)} frame(s)."],
                actual_signal=getattr(visual_matches[0], 'text', '')[:100] if visual_matches else None,
            )

        # ─────────────────────────────────────────────────────────────────────
        # 3. Log confirms SUCCESS without visual evidence → Observed (trust browser)
        # ─────────────────────────────────────────────────────────────────────
        if log_confirmed_success:
            return ClassificationResult(
                status=StepStatus.OBSERVED,
                confidence=0.75,  # Slightly lower confidence than visual
                notes=["Browser logs confirm successful execution (no visual verification available)."],
            )

        # ─────────────────────────────────────────────────────────────────────
        # 4. Hallucination detection (HALoGEN principle)
        #    Log claims execution but no confirmation of success
        # ─────────────────────────────────────────────────────────────────────
        if log_has_claim and not visual_matches:
            return ClassificationResult(
                status=StepStatus.HALLUCINATED,
                confidence=0.7,
                notes=["Log claims step was executed but no visual evidence found."],
            )

        # ─────────────────────────────────────────────────────────────────────
        # 4. No strong evidence → Unclear
        # ─────────────────────────────────────────────────────────────────────
        if not visual_matches and not log_has_claim:
            has_partial = self._check_partial_evidence(step, evidence)
            if has_partial:
                return ClassificationResult(
                    status=StepStatus.UNCLEAR,
                    confidence=0.4,
                    notes=["Partial evidence found; manual review recommended."],
                )
            return ClassificationResult(
                status=StepStatus.UNCLEAR,
                confidence=0.3,
                notes=["Insufficient evidence to classify."],
            )

        # Default fallback
        return ClassificationResult(
            status=StepStatus.UNCLEAR,
            confidence=0.2,
            notes=["Classification uncertain."],
        )

    def _check_partial_evidence(
        self,
        step: "PlanStep",
        evidence: "EvidenceStore",
    ) -> bool:
        """Check if there's any partial evidence for the step."""
        action = getattr(step, 'action', '').lower()
        if not action:
            return False

        all_items = list(getattr(evidence, 'screenshots', [])) + \
                    list(getattr(evidence, 'video_frames', []))

        for item in all_items:
            text = getattr(item, 'text', '').lower()
            if action in text:
                return True
        return False

    def detect_hallucination(
        self,
        step: "PlanStep",
        log_claims: list[str],
        visual_evidence: list,
    ) -> tuple[bool, str]:
        """
        Detect if a step was hallucinated (claimed in logs but not observed).

        Based on HALoGEN research: systematically identify when models
        claim things that did not happen.

        Returns:
            (is_hallucinated, explanation)
        """
        action = getattr(step, 'action', '').lower()

        # Check if logs claim this action was performed
        action_claimed = False
        claim_text = ""

        keywords = self.ACTION_KEYWORDS.get(action, [action])
        for claim in log_claims:
            claim_lower = claim.lower()
            if any(kw in claim_lower for kw in keywords):
                action_claimed = True
                claim_text = claim
                break

        if not action_claimed:
            return False, "No log claim found for this action."

        # Check if visual evidence supports the claim
        if action_claimed and not visual_evidence:
            explanation = (
                f"Log claims '{claim_text[:50]}...' but no visual evidence "
                "found in video frames or screenshots."
            )
            return True, explanation

        return False, "Log claim is supported by visual evidence."

    def calculate_confidence(
        self,
        visual_matches: int,
        log_support: bool,
        previous_step_passed: bool,
        action_type: str,
    ) -> float:
        """
        Calculate confidence score for classification.

        Factors (BrowseComp-inspired):
        - Number of visual matches: +0.15 per match (max 0.6)
        - Log support: +0.2
        - Previous step passed: +0.1
        - Action type reliability: +0.1 for high-reliability actions
        """
        confidence = 0.0

        # Visual evidence (up to 4 matches)
        confidence += min(0.6, visual_matches * 0.15)

        # Log support
        if log_support:
            confidence += 0.2

        # Previous step success (build momentum)
        if previous_step_passed:
            confidence += 0.1

        # Action type reliability (some actions are easier to verify)
        high_reliability_actions = {"navigate", "openurl", "click"}
        if action_type.lower() in high_reliability_actions:
            confidence += 0.1

        return min(1.0, confidence)


def create_classifier() -> DeviationClassifier:
    """Factory function to create a classifier instance."""
    return DeviationClassifier()
