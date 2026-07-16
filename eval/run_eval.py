"""
Evaluation harness — measures agent performance against ground truth.

Reports:
  - Status classification precision/recall
  - Insufficiency detection F1
  - Follow-up grouping accuracy
  - Cost per PBC list processed
"""

import json
import sys
from pathlib import Path


def load_ground_truth(path: str) -> dict:
    """Load the ground truth JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_statuses(predicted: dict[str, str], expected: dict[str, dict]) -> dict:
    """
    Evaluate status classification accuracy.
    Returns precision, recall, F1 for each status category.
    """
    categories = ["Not started", "Received", "Insufficient", "Under review", "Complete"]
    results = {}

    for category in categories:
        tp = fp = fn = 0
        for item_id, expected_data in expected.items():
            expected_status = expected_data["status"]
            predicted_status = predicted.get(item_id, "Not started")

            if predicted_status == category and expected_status == category:
                tp += 1
            elif predicted_status == category and expected_status != category:
                fp += 1
            elif predicted_status != category and expected_status == category:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[category] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Overall accuracy
    correct = sum(
        1 for item_id, data in expected.items()
        if predicted.get(item_id, "Not started") == data["status"]
    )
    results["overall_accuracy"] = round(correct / len(expected), 3)

    return results


def evaluate_insufficiency_detection(predicted: dict[str, str], expected: dict[str, dict]) -> dict:
    """
    Specifically evaluate insufficiency detection — the hardest task.
    """
    tp = fp = fn = 0
    details = []

    for item_id, expected_data in expected.items():
        is_insufficient_expected = expected_data["status"] == "Insufficient"
        is_insufficient_predicted = predicted.get(item_id, "Not started") == "Insufficient"

        if is_insufficient_predicted and is_insufficient_expected:
            tp += 1
            details.append({"item": item_id, "result": "TP"})
        elif is_insufficient_predicted and not is_insufficient_expected:
            fp += 1
            details.append({"item": item_id, "result": "FP", "expected": expected_data["status"]})
        elif not is_insufficient_predicted and is_insufficient_expected:
            fn += 1
            details.append({"item": item_id, "result": "FN", "predicted": predicted.get(item_id, "Not started")})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "details": details,
    }


def evaluate_followups(predicted_followups: list[dict], expected_followups: list[dict]) -> dict:
    """Evaluate follow-up email grouping."""
    results = {
        "expected_groups": len(expected_followups),
        "predicted_groups": len(predicted_followups),
        "recipient_matches": 0,
        "item_coverage": 0.0,
    }

    for expected_group in expected_followups:
        expected_email = expected_group["recipient"]
        expected_items = set(expected_group["items"])

        # Find matching predicted group
        for pred_group in predicted_followups:
            if pred_group.get("recipient_email", "") == expected_email:
                results["recipient_matches"] += 1
                pred_items = set(pred_group.get("items_covered", []))
                overlap = expected_items & pred_items
                results["item_coverage"] += len(overlap) / len(expected_items)
                break

    if expected_followups:
        results["item_coverage"] = round(results["item_coverage"] / len(expected_followups), 3)

    return results


def run_evaluation(run_result: dict, ground_truth_path: str) -> dict:
    """
    Run full evaluation suite.
    """
    gt = load_ground_truth(ground_truth_path)
    expected_statuses = gt["expected_status"]
    expected_followups = gt.get("expected_followup_groups", [])

    # Extract predicted statuses from run result
    predicted_statuses = {}
    for item_id, item_data in run_result.get("items", {}).items():
        predicted_statuses[item_id] = item_data.get("status", "Not started")

    # Run evaluations
    status_eval = evaluate_statuses(predicted_statuses, expected_statuses)
    insufficiency_eval = evaluate_insufficiency_detection(predicted_statuses, expected_statuses)
    followup_eval = evaluate_followups(
        run_result.get("followup_drafts", []),
        expected_followups,
    )

    report = {
        "status_classification": status_eval,
        "insufficiency_detection": insufficiency_eval,
        "followup_grouping": followup_eval,
    }

    return report


def print_report(report: dict, cost_usd: float = 0.0) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("PBC AGENT EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'='*60}")
    print("STATUS CLASSIFICATION")
    print("-" * 60)
    print(f"  Overall Accuracy: {report['status_classification']['overall_accuracy']:.1%}")
    for status, metrics in report['status_classification'].items():
        if status == 'overall_accuracy':
            continue
        if isinstance(metrics, dict) and metrics.get('tp', 0) + metrics.get('fp', 0) + metrics.get('fn', 0) > 0:
            print(f"  {status:15s} — P: {metrics['precision']:.2f}  R: {metrics['recall']:.2f}  F1: {metrics['f1']:.2f}")

    print(f"\n{'='*60}")
    print("INSUFFICIENCY DETECTION")
    print("-" * 60)
    insuff = report['insufficiency_detection']
    print(f"  Precision: {insuff['precision']:.2f}")
    print(f"  Recall:    {insuff['recall']:.2f}")
    print(f"  F1:        {insuff['f1']:.2f}")
    if insuff.get('details'):
        for d in insuff['details']:
            print(f"    {d['item']}: {d['result']}" + (f" (expected {d.get('expected', d.get('predicted', ''))})" if d['result'] != 'TP' else ''))

    print(f"\n{'='*60}")
    print("FOLLOW-UP GROUPING")
    print("-" * 60)
    fu = report['followup_grouping']
    print(f"  Expected groups:   {fu['expected_groups']}")
    print(f"  Predicted groups:  {fu['predicted_groups']}")
    print(f"  Recipient matches: {fu['recipient_matches']}")
    print(f"  Item coverage:     {fu['item_coverage']:.1%}")

    print(f"\n{'='*60}")
    print("COST")
    print("-" * 60)
    print(f"  Total cost: ${cost_usd:.4f}")
    print(f"  Budget:     $2.00")
    print(f"  Under budget: {'✓' if cost_usd <= 2.0 else '✗ OVER BUDGET'}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage with saved run results
    if len(sys.argv) < 3:
        print("Usage: python -m eval.run_eval <run_results.json> <ground_truth.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        run_result = json.load(f)
    report = run_evaluation(run_result, sys.argv[2])
    print_report(report, cost_usd=run_result.get("cost_usd", 0))
