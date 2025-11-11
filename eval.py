import json
import argparse
import sys
from typing import List, Dict, Any, Optional

def load_json_as_dict(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the JSON list output from your pipeline and converts it
    to a dictionary keyed by 'field_name' for easy lookup.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle both list output (from ExtractionOutputs) and flat dict
            if isinstance(data, list):
                # Handle cases where a field might be duplicated (e.g., in different pages)
                # For this script, we'll just take the first one found.
                processed_dict = {}
                for item in data:
                    if "field_name" in item and item["field_name"] not in processed_dict:
                        processed_dict[item["field_name"]] = item
                return processed_dict
            elif isinstance(data, dict):
                # If it's already a flat dict, just return it
                return data
            else:
                print(f"Error: Unknown JSON structure in {filepath}. Expected list or dict.")
                sys.exit(1)
                
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def normalize_text(val: Any) -> str:
    """
    Converts any value to a normalized string for comparison.
    - Strips whitespace
    - Converts to lowercase
    - Handles None
    """
    if val is None:
        return ""
    return str(val).strip().lower()

def compare_values(pred_val: Any, gt_val: Any, field_type: str) -> bool:
    """
    Compares predicted and ground truth values based on field type.
    Handles all specified types: key-value, summary, checkbox, table, bullet-points.
    """
    # Normalize 'bullet-points' to 'bullet_points'
    field_type = field_type.replace('-', '_')

    if field_type == "key_value":
        # For key-value pairs, compare normalized text.
        return normalize_text(pred_val) == normalize_text(gt_val)
        
    elif field_type == "summary":
        # For summaries, compare normalized text.
        # A more advanced check would use semantic similarity (e.g., ROUGE, BERTScore).
        return normalize_text(pred_val) == normalize_text(gt_val)
        
    elif field_type == "checkbox":
        # For checkboxes, we can compare normalized text (e.g., "checked", "true", "on")
        # or direct booleans. Normalized text is safer.
        return normalize_text(pred_val) == normalize_text(gt_val)

    elif field_type == "table":
        # For tables, we expect a list of dicts.
        # This does a simple, order-sensitive comparison.
        # A more advanced check would compare rows/cells more robustly.
        return pred_val == gt_val

    elif field_type == "bullet_points":
        # For bullet points, we also expect a list.
        # This does a simple, order-sensitive comparison.
        return pred_val == gt_val
        
    else: # Default case for any other types
        # Fallback to normalized text comparison
        print(f"Warning: Unknown field_type '{field_type}'. Using text comparison.")
        return normalize_text(pred_val) == normalize_text(gt_val)

def evaluate(pred_file: str, gt_file: str):
    """
    Main evaluation function.
    Compares a predicted JSON file against a ground truth JSON file.
    """
    print(f"Loading predicted data from: {pred_file}")
    predicted_data = load_json_as_dict(pred_file)
    
    print(f"Loading ground truth data from: {gt_file}")
    ground_truth_data = load_json_as_dict(gt_file)

    if not ground_truth_data:
        print("Error: Ground truth file is empty or invalid.")
        return

    all_field_names = sorted(list(set(predicted_data.keys()) | set(ground_truth_data.keys())))
    
    results = {
        "correct": 0,
        "incorrect_value": 0,
        "missed_field_fn": 0, # False Negative (in GT, not in Pred)
        "extra_field_fp": 0   # False Positive (in Pred, not in GT)
    }
    
    print("\n" + "="*30)
    print("FIELD-BY-FIELD EVALUATION")
    print("="*30)

    for field_name in all_field_names:
        in_pred = field_name in predicted_data
        in_gt = field_name in ground_truth_data
        
        if in_pred and in_gt:
            # Field was found (True Positive)
            pred_item = predicted_data[field_name]
            gt_item = ground_truth_data[field_name]
            
            # Use 'type' from GT file, default to 'key_value'
            field_type = gt_item.get("type", "key_value")
            
            # Use 'value' key, fallback to 'item' if 'value' isn't present
            pred_val = pred_item.get("value", pred_item) 
            gt_val = gt_item.get("value", gt_item)
            
            if compare_values(pred_val, gt_val, field_type):
                print(f"[ CORRECT ] {field_name}")
                results["correct"] += 1
            else:
                print(f"[INCORRECT] {field_name}")
                print(f"  - GT:   {str(gt_val)[:100]}...")
                print(f"  - PRED: {str(pred_val)[:100]}...")
                results["incorrect_value"] += 1
                
        elif in_gt:
            # Field was missed (False Negative)
            print(f"[  MISSED ] {field_name} (In Ground Truth, not in Output)")
            results["missed_field_fn"] += 1
            
        elif in_pred:
            # Field was extra (False Positive)
            print(f"[  EXTRA  ] {field_name} (In Output, not in Ground Truth)")
            results["extra_field_fp"] += 1

    # --- Print Summary ---
    total_gt_fields = len(ground_truth_data)
    
    # Accuracy = Correct / (Correct + Incorrect + Missed)
    denominator = results["correct"] + results["incorrect_value"] + results["missed_field_fn"]
    if denominator == 0:
        accuracy = 0.0
    else:
        accuracy = (results["correct"] / denominator) * 100
        
    precision_denominator = results['correct'] + results['extra_field_fp']
    precision = (results['correct'] / precision_denominator * 100) if precision_denominator > 0 else 0.0

    recall_denominator = results['correct'] + results['missed_field_fn']
    recall = (results['correct'] / recall_denominator * 100) if recall_denominator > 0 else 0.0


    print("\n" + "="*30)
    print("OVERALL RESULTS")
    print("="*30)
    print(f"Total Fields in Ground Truth: {total_gt_fields}")
    print("-------------------------")
    print(f"Correct Fields:             {results['correct']}")
    print(f"Incorrect Values:           {results['incorrect_value']}")
    print(f"Missed Fields (FN):         {results['missed_field_fn']}")
    print(f"Extra Fields (FP):          {results['extra_field_fp']}")
    print("-------------------------")
    print(f"Accuracy (Correct / (Correct + Incorrect + Missed)): {accuracy:.2f}%")
    print(f"Precision (Correct / (Correct + Extra)): {precision:.2f}%")
    print(f"Recall (Correct / (Correct + Missed)): {recall:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate extraction output against ground truth.")
    parser.add_argument(
        "--pred", 
        required=True, 
        help="Path to the predicted JSON file (your pipeline's output)."
    )
    parser.add_argument(
        "--gt", 
        required=True, 
        help="Path to the ground truth JSON file."
    )
    
    args = parser.parse_args()
    evaluate(args.pred, args.gt)
