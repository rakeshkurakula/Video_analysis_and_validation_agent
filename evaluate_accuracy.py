"""Evaluation framework for deviation classification accuracy based on FRAMES research."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationMetrics:
    """Metrics for classification accuracy."""
    total_steps: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    precision_by_class: dict[str, float] = field(default_factory=dict)
    recall_by_class: dict[str, float] = field(default_factory=dict)
    f1_by_class: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy, 4),
            "precision_by_class": {k: round(v, 4) for k, v in self.precision_by_class.items()},
            "recall_by_class": {k: round(v, 4) for k, v in self.recall_by_class.items()},
            "f1_by_class": {k: round(v, 4) for k, v in self.f1_by_class.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class StepComparison:
    """Comparison between predicted and gold step classification."""
    step_index: int
    description: str
    predicted_status: str
    gold_status: str
    correct: bool
    predicted_confidence: float = 0.0
    notes: str = ""


class AccuracyEvaluator:
    """
    Evaluate predicted deviation reports against gold-labeled ground truth.
    
    Based on FRAMES and BrowseComp research for structured evaluation.
    """
    
    # Status normalization mapping
    STATUS_ALIASES = {
        "observed": "Observed",
        "deviation": "Deviation",
        "deviation-skipped": "Deviation-Skipped",
        "deviation-altered": "Deviation-Altered",
        "hallucinated": "Hallucinated",
        "unclear": "Unclear",
        "partially-observed": "Partially-Observed",
        "skipped": "Deviation-Skipped",
        "altered": "Deviation-Altered",
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize evaluator.
        
        Args:
            strict_mode: If True, require exact status match.
                        If False, treat some statuses as equivalent.
        """
        self.strict_mode = strict_mode
    
    def evaluate(
        self,
        predicted_report: dict | Path,
        gold_report: dict | Path,
    ) -> EvaluationMetrics:
        """
        Compare predicted classifications against gold labels.
        
        Args:
            predicted_report: Predicted deviation report (dict or path to JSON)
            gold_report: Gold-labeled ground truth (dict or path to JSON)
        
        Returns:
            EvaluationMetrics with accuracy, precision, recall, F1
        """
        if isinstance(predicted_report, Path):
            predicted_report = json.loads(predicted_report.read_text())
        if isinstance(gold_report, Path):
            gold_report = json.loads(gold_report.read_text())
        
        predicted_steps = self._extract_steps(predicted_report)
        gold_steps = self._extract_steps(gold_report)
        
        # Align steps by index
        comparisons = []
        all_statuses = set()
        
        for idx in sorted(set(predicted_steps.keys()) | set(gold_steps.keys())):
            pred = predicted_steps.get(idx, {"status": "Missing", "description": ""})
            gold = gold_steps.get(idx, {"status": "Missing", "description": ""})
            
            pred_status = self._normalize_status(pred.get("status", "Missing"))
            gold_status = self._normalize_status(gold.get("status", "Missing"))
            
            all_statuses.add(pred_status)
            all_statuses.add(gold_status)
            
            correct = self._statuses_match(pred_status, gold_status)
            
            comparisons.append(StepComparison(
                step_index=idx,
                description=pred.get("description", gold.get("description", "")),
                predicted_status=pred_status,
                gold_status=gold_status,
                correct=correct,
                predicted_confidence=pred.get("confidence", 0.0),
            ))
        
        # Calculate metrics
        metrics = self._calculate_metrics(comparisons, list(all_statuses))
        return metrics
    
    def _extract_steps(self, report: dict) -> dict[int, dict]:
        """Extract steps from report, keyed by index."""
        steps = {}
        for step in report.get("steps", []):
            idx = step.get("index", 0)
            steps[idx] = step
        return steps
    
    def _normalize_status(self, status: str) -> str:
        """Normalize status string to canonical form."""
        lower = status.lower().strip()
        return self.STATUS_ALIASES.get(lower, status)
    
    def _statuses_match(self, predicted: str, gold: str) -> bool:
        """Check if statuses match (with optional leniency)."""
        if predicted == gold:
            return True
        
        if self.strict_mode:
            return False
        
        # Non-strict: treat some statuses as equivalent
        equivalent_groups = [
            {"Deviation", "Deviation-Skipped", "Deviation-Altered"},
            {"Unclear", "Missing"},
        ]
        
        for group in equivalent_groups:
            if predicted in group and gold in group:
                return True
        
        return False
    
    def _calculate_metrics(
        self,
        comparisons: list[StepComparison],
        all_statuses: list[str],
    ) -> EvaluationMetrics:
        """Calculate precision, recall, F1, and confusion matrix."""
        metrics = EvaluationMetrics()
        metrics.total_steps = len(comparisons)
        metrics.correct_predictions = sum(1 for c in comparisons if c.correct)
        metrics.accuracy = (
            metrics.correct_predictions / metrics.total_steps
            if metrics.total_steps > 0 else 0.0
        )
        
        # Build confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for c in comparisons:
            confusion[c.gold_status][c.predicted_status] += 1
        metrics.confusion_matrix = {k: dict(v) for k, v in confusion.items()}
        
        # Calculate per-class metrics
        for status in all_statuses:
            # True positives: predicted and gold both = status
            tp = sum(
                1 for c in comparisons
                if c.predicted_status == status and c.gold_status == status
            )
            
            # False positives: predicted = status but gold != status
            fp = sum(
                1 for c in comparisons
                if c.predicted_status == status and c.gold_status != status
            )
            
            # False negatives: gold = status but predicted != status
            fn = sum(
                1 for c in comparisons
                if c.gold_status == status and c.predicted_status != status
            )
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics.precision_by_class[status] = precision
            
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics.recall_by_class[status] = recall
            
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            metrics.f1_by_class[status] = f1
        
        return metrics
    
    def generate_comparison_report(
        self,
        predicted_report: dict | Path,
        gold_report: dict | Path,
        output_path: Path | None = None,
    ) -> str:
        """Generate a detailed comparison report in markdown format."""
        if isinstance(predicted_report, Path):
            predicted_report = json.loads(predicted_report.read_text())
        if isinstance(gold_report, Path):
            gold_report = json.loads(gold_report.read_text())
        
        predicted_steps = self._extract_steps(predicted_report)
        gold_steps = self._extract_steps(gold_report)
        
        lines = [
            "# Accuracy Evaluation Report",
            "",
            f"- Scenario: {predicted_report.get('scenario', 'Unknown')}",
            f"- Run ID: {predicted_report.get('run_id', 'Unknown')}",
            "",
        ]
        
        # Calculate metrics
        metrics = self.evaluate(predicted_report, gold_report)
        
        lines.extend([
            "## Summary Metrics",
            "",
            f"- **Accuracy**: {metrics.accuracy:.1%}",
            f"- **Correct/Total**: {metrics.correct_predictions}/{metrics.total_steps}",
            "",
        ])
        
        # Per-class breakdown
        lines.append("## Per-Class Metrics")
        lines.append("")
        lines.append("| Status | Precision | Recall | F1 |")
        lines.append("| --- | --- | --- | --- |")
        
        for status in sorted(metrics.precision_by_class.keys()):
            p = metrics.precision_by_class.get(status, 0)
            r = metrics.recall_by_class.get(status, 0)
            f1 = metrics.f1_by_class.get(status, 0)
            lines.append(f"| {status} | {p:.1%} | {r:.1%} | {f1:.2f} |")
        
        # Step-by-step comparison
        lines.append("")
        lines.append("## Step-by-Step Comparison")
        lines.append("")
        lines.append("| Step | Description | Predicted | Gold | Match |")
        lines.append("| --- | --- | --- | --- | --- |")
        
        for idx in sorted(set(predicted_steps.keys()) | set(gold_steps.keys())):
            pred = predicted_steps.get(idx, {})
            gold = gold_steps.get(idx, {})
            
            pred_status = self._normalize_status(pred.get("status", "Missing"))
            gold_status = self._normalize_status(gold.get("status", "Missing"))
            match = "✅" if self._statuses_match(pred_status, gold_status) else "❌"
            desc = pred.get("description", gold.get("description", ""))[:50]
            
            lines.append(f"| {idx} | {desc} | {pred_status} | {gold_status} | {match} |")
        
        report_text = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report_text, encoding="utf-8")
        
        return report_text


def create_gold_label_template(
    predicted_report: dict | Path,
    output_path: Path,
) -> None:
    """
    Create a gold label template from a predicted report.
    
    The template can be edited manually to create ground truth labels.
    """
    if isinstance(predicted_report, Path):
        predicted_report = json.loads(predicted_report.read_text())
    
    template = {
        "scenario": predicted_report.get("scenario", ""),
        "run_id": predicted_report.get("run_id", ""),
        "steps": [],
    }
    
    for step in predicted_report.get("steps", []):
        template["steps"].append({
            "index": step.get("index"),
            "description": step.get("description"),
            "status": step.get("status"),  # Copy predicted as starting point
            "notes": "REVIEW: verify this classification manually",
        })
    
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def evaluate_accuracy(
    predicted_path: Path,
    gold_path: Path,
    strict: bool = False,
) -> EvaluationMetrics:
    """Convenience function to evaluate accuracy."""
    evaluator = AccuracyEvaluator(strict_mode=strict)
    return evaluator.evaluate(predicted_path, gold_path)
