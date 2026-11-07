# MLOps CI/CD/CT & Governance: Multi-metric validation, regression margin guardrails, and automated promotion decisions

import json
from typing import Any, Dict, List, Optional, Tuple


class EvaluationMetrics:
    """Computes binary classification performance metrics from raw predictions and labels."""
    def __init__(self, y_true: List[int], y_pred: List[int]):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have identical lengths.")
        if not y_true:
            raise ValueError("Datasets must not be empty.")

        self.tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        self.fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        self.tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        self.fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    def calculate_all(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4)
        }


class ModelEvaluationGate:
    """
    Evaluates candidate models against explicit minimum thresholds and baseline
    regression margins to determine safe CI/CD pipeline promotion.
    """
    def __init__(
        self,
        min_thresholds: Dict[str, float],
        max_regression_delta: float = 0.02
    ):
        self.min_thresholds = min_thresholds
        self.max_regression_delta = max_regression_delta

    def evaluate_candidate(
        self,
        candidate_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validates candidate against static floors and baseline performance.
        Returns: (passed: bool, failure_reasons: List[str], audit_report: Dict)
        """
        violations = []
        checks = {}

        for metric_name, floor_val in self.min_thresholds.items():
            actual_val = candidate_metrics.get(metric_name)
            if actual_val is None:
                violations.append(f"Required metric '{metric_name}' missing from candidate metrics.")
                continue

            passed_check = actual_val >= floor_val
            checks[f"floor_{metric_name}"] = {
                "required": floor_val,
                "actual": actual_val,
                "passed": passed_check
            }
            if not passed_check:
                violations.append(
                    f"Threshold failed: {metric_name} ({actual_val}) < minimum required ({floor_val})"
                )
        if baseline_metrics:
            for metric_name, base_val in baseline_metrics.items():
                if metric_name in candidate_metrics:
                    cand_val = candidate_metrics[metric_name]
                    delta = base_val - cand_val 

                    regressed = delta > self.max_regression_delta
                    checks[f"regression_{metric_name}"] = {
                        "baseline": base_val,
                        "candidate": cand_val,
                        "drop_delta": round(delta, 4),
                        "passed": not regressed
                    }
                    if regressed:
                        violations.append(
                            f"Regression detected: {metric_name} dropped by {round(delta, 4)} "
                            f"(exceeds allowed margin {self.max_regression_delta})"
                        )

        passed = len(violations) == 0
        audit_report = {
            "passed": passed,
            "violations": violations,
            "checks": checks,
            "candidate_metrics": candidate_metrics
        }
        return passed, violations, audit_report


if __name__ == "__main__":
    print("--- MLOps Governance: Automated Model Evaluation Gate ---\n")

    # Minimum acceptable performance floors for fraud detection
    production_thresholds = {
        "recall": 0.85,
        "f1_score": 0.80,
        "accuracy": 0.88
    }

    gate = ModelEvaluationGate(
        min_thresholds=production_thresholds,
        max_regression_delta=0.02
    )

    baseline_prod_metrics = {
        "accuracy": 0.9100,
        "precision": 0.8800,
        "recall": 0.8700,
        "f1_score": 0.8750
    }

    print("Scenario 1: Evaluating Candidate Model A (High Performance):")
    ground_truth = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    preds_candidate_a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

    metrics_a = EvaluationMetrics(ground_truth, preds_candidate_a).calculate_all()
    passed_a, violations_a, report_a = gate.evaluate_candidate(metrics_a, baseline_prod_metrics)

    print(f"  Candidate Metrics: {metrics_a}")
    print(f"  Gate Verdict: {'APPROVED' if passed_a else 'REJECTED'}")
    if violations_a:
        print(f"  Violations: {violations_a}")

    print("\nScenario 2: Evaluating Candidate Model B (Degraded Recall):")
    preds_candidate_b = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]

    metrics_b = EvaluationMetrics(ground_truth, preds_candidate_b).calculate_all()
    passed_b, violations_b, report_b = gate.evaluate_candidate(metrics_b, baseline_prod_metrics)

    print(f"  Candidate Metrics: {metrics_b}")
    print(f"  Gate Verdict: {'APPROVED' if passed_b else 'REJECTED'}")
    print(f"  Violations: {violations_b}")

    print("\nFull Gate Audit Report (Scenario 2):")
    print(json.dumps(report_b, indent=2))