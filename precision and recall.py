def evaluate_detection_and_classification(ground_truths, predictions):
    tp, fp, fn = 0, 0, 0
    gt_ids_and_genders = {(gt['id'], gt['gender']) for gt in ground_truths}
    pred_ids_and_genders = {(pred['id'], pred['gender']) for pred in predictions}

    # Calculate True Positives and False Positives
    matched_preds = set()
    for gt in ground_truths:
        if (gt['id'], gt['gender']) in pred_ids_and_genders:
            tp += 1
            matched_preds.add((gt['id'], gt['gender']))
    
    fp = len(pred_ids_and_genders) - len(matched_preds)
    fn = len(gt_ids_and_genders) - len(matched_preds)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

# Example ground truth and predictions
ground_truths = [
    {'id': 1, 'frame': 1, 'gender': 'female'},
    {'id': 2, 'frame': 1, 'gender': 'male'},
    {'id': 3, 'frame': 1, 'gender': 'male'},
    {'id': 4, 'frame': 1, 'gender': 'female'},
    {'id': 5, 'frame': 1, 'gender': 'female'},
    {'id': 6, 'frame': 1, 'gender': 'male'},
    {'id': 7, 'frame': 1, 'gender': 'female'},
    {'id': 8, 'frame': 1, 'gender': 'male'},
    {'id': 9, 'frame': 1, 'gender': 'female'},
    {'id': 10, 'frame':1, 'gender': 'male'}
   
    # Add more ground truth annotations
]

predictions =[

    {'id': 1, 'frame': 1, 'gender': 'male'},
    {'id': 2, 'frame': 1, 'gender': 'male'},
    {'id': 3, 'frame': 1, 'gender': 'male'},
    {'id': 4, 'frame': 1, 'gender': 'male'},
    {'id': 5, 'frame': 1, 'gender': 'male'},
    {'id': 6, 'frame': 1, 'gender': 'male'},
    {'id': 7, 'frame': 1, 'gender': 'female'},
    {'id': 8, 'frame': 1, 'gender': 'female'},
    {'id': 9, 'frame': 1, 'gender': 'male'},
    {'id': 10, 'frame':1, 'gender': 'male'},


    # Add more predictions
]

precision, recall = evaluate_detection_and_classification(ground_truths, predictions)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
