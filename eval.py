import json
import argparse
import sys
from typing import List, Dict, Any, Optional

# Try to import thefuzz for fuzzy matching
try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
except ImportError:
    print("Error: 'thefuzz' library not found. Please run: pip install thefuzz python-Levenshtein")
    sys.exit(1)

# --- CONFIGURATION ---
FUZZY_THRESHOLD = 90  # Score out of 100. If similarity > 90, we consider it correct.

def load_json_as_dict(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the JSON output and converts it to a dictionary keyed by 'field_name'.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                processed_dict = {}
                for item in data:
                    # Normalize field_name to string to avoid errors
                    fname = str(item.get("field_name", ""))
                    if fname and fname not in processed_dict:
                        processed_dict[fname] = item
                return processed_dict
            elif isinstance(data, dict):
                return data
            else:
                print(f"Error: Unknown JSON structure in {filepath}.")
                sys.exit(1)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def normalize_text(val: Any) -> str:
    """
    Converts values to normalized strings (lowercase, stripped) for comparison.
    """
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        return str(val).lower()
    return str(val).strip().lower()

def check_text_similarity(pred: str, gt: str, method: str) -> tuple[bool, int, str]:
    """
    Helper to check similarity based on the chosen method.
    Returns: (passed_bool, score_int, match_type_str)
    """
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    # 1. Exact Match (always check this first)
    if pred_norm == gt_norm:
        return True, 100, "Exact"

    # 2. If method is 'exact', and it failed above, it's a hard fail.
    if method == "exact":
        return False, 0, "Fail (Exact)"

    # 3. If method is 'fuzzy', calculate fuzzy score
    if method == "fuzzy":
        score = fuzz.token_sort_ratio(pred_norm, gt_norm)
        if score >= FUZZY_THRESHOLD:
            return True, score, "Fuzzy"
        else:
            return False, score, f"Fail (Fuzzy: {score}%)"
    
    # Default fail
    return False, 0, "Fail"


def compare_values(pred_val: Any, gt_val: Any, field_type: str, method: str) -> tuple[bool, str]:
    """
    Compares values and returns (Passed?, Message).
    """
    field_type = str(field_type).replace('-', '_') # normalize bullet-points -> bullet_points

    # --- TEXT FIELDS (Key-Value, Summary, Checkbox) ---
    if field_type in ["key_value", "summary", "checkbox"]:
        passed, score, match_type = check_text_similarity(str(pred_val), str(gt_val), method)
        if passed:
            return True, f"Match ({match_type}: {score}%)"
        else:
            return False, f"Score: {score}% (Threshold: {FUZZY_THRESHOLD}%)"

    # --- LIST FIELDS (Table, Bullet Points) ---
    elif field_type in ["table", "bullet_points"]:
        if not isinstance(pred_val, list) or not isinstance(gt_val, list):
             return False, "Type Mismatch (Expected List)"
             
        if len(pred_val) != len(gt_val):
            return False, f"Length Mismatch (GT: {len(gt_val)}, Pred: {len(pred_val)})"

        for i, (p_item, g_item) in enumerate(zip(pred_val, gt_val)):
            p_str = str(p_item)
            g_str = str(g_item)
                
            passed, score, match_type = check_text_similarity(p_str, g_str, method)
            if not passed:
                if field_type == "table":
                    return False, f"Row {i+1} mismatch ({match_type}: {score}%)"
                else:
                    return False, f"Item {i+1} mismatch ({match_type}: {score}%)"

        return True, "Match (List Content)"
        
    # --- UNKNOWN TYPES ---
    else:
        passed, score, match_type = check_text_similarity(str(pred_val), str(gt_val), method)
        return passed, f"Unknown Type ({match_type}: {score}%)"

def evaluate(pred_file: str, gt_file: str, method: str):
    print(f"Loading prediction:   {pred_file}")
    print(f"Loading ground truth: {gt_file}")
    print(f"Running evaluation in: '{method.upper()}' mode")
    if method == 'fuzzy':
        print(f"Fuzzy Threshold set to: {FUZZY_THRESHOLD}%")

    
    pred_data = load_json_as_dict(pred_file)
    gt_data = load_json_as_dict(gt_file)

    if not gt_data:
        print("Error: Ground truth is empty.")
        return

    all_fields = sorted(list(set(pred_data.keys()) | set(gt_data.keys())))
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "missed": 0,
        "extra": 0
    }

    print("\n" + "="*60)
    print(f"{'FIELD NAME':<30} | {'STATUS':<10} | {'DETAILS'}")
    print("="*60)

    for field in all_fields:
        in_p = field in pred_data
        in_g = field in gt_data
        
        if in_p and in_g:
            p_item = pred_data[field]
            g_item = gt_data[field]
            
            f_type = g_item.get("type", "key_value")
            p_val = p_item.get("value", p_item)
            g_val = g_item.get("value", g_item)
            
            passed, msg = compare_values(p_val, g_val, f_type, method)
            
            if passed:
                status = "CORRECT"
                results["correct"] += 1
                print(f"{field:<30} | {status:<10} | {msg}")
            else:
                status = "FAIL"
                results["incorrect"] += 1
                print(f"{field:<30} | {status:<10} | {msg}")
                print(f"   GT:   {str(g_val)[:80]}...")
                print(f"   PRED: {str(p_val)[:80]}...")

        elif in_g:
            print(f"{field:<30} | MISSED     | Not in output")
            results["missed"] += 1
        elif in_p:
            print(f"{field:<30} | EXTRA      | Not in ground truth")
            results["extra"] += 1

    # --- SUMMARY ---
    total_gt_fields = len(gt_data)
    total_correct = results["correct"]
    total_errors = results["incorrect"] + results["missed"]
    
    print("\n" + "="*30)
    print(f"EVALUATION SUMMARY ({method.upper()})")
    print("="*30)
    print(f"Total Fields in GT: {total_gt_fields}")
    print(f"Correct Fields:     {results['correct']}")
    print(f"Incorrect Values:   {results['incorrect']}")
    print(f"Missed Fields (FN): {results['missed']}")
    print(f"Extra Fields (FP):  {results['extra']}")
    print("-" * 30)
    
    # Accuracy = Correct / (Correct + Incorrect + Missed)
    denominator_acc = total_correct + results['incorrect'] + results['missed']
    if denominator_acc > 0:
        acc = (total_correct / denominator_acc) * 100
        print(f"Accuracy: {acc:.2f}% (Correct / (Correct + Incorrect + Missed))")
    else:
        print("Accuracy: N/A")

    # Precision = Correct / (Correct + Extra)
    denominator_prec = total_correct + results['extra']
    if denominator_prec > 0:
        prec = (total_correct / denominator_prec) * 100
        print(f"Precision: {prec:.2f}% (Correct / (Correct + Extra))")
    else:
        print("Precision: N/A")

    # Recall = Correct / (Correct + Missed)
    denominator_recall = total_correct + results['missed']
    if denominator_recall > 0:
        recall = (total_correct / denominator_recall) * 100
        print(f"Recall: {recall:.2f}% (Correct / (Correct + Missed))")
    else:
        print("Recall: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Predicted JSON file")
    parser.add_argument("--gt", required=True, help="Ground Truth JSON file")
    parser.add_argument(
        "--method",
        required=True,
        choices=['exact', 'fuzzy'],
        help="Evaluation method: 'exact' (strict) or 'fuzzy' (flexible)."
    )
    args = parser.parse_args()
    
    evaluate(args.pred, args.gt, args.method)
