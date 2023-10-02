import json
import numpy as np
from sklearn.metrics import precision_score, recall_score

head_preds = "./pred/all-deberta-large-nlu-entity-test_head/predictions.json"
tail_preds = "./pred/all-deberta-large-nlu-entity-test_tail/predictions.json"
head_labels = "./data/all/entity/nlu/test_head/labels.json"
tail_labels = "./data/all/entity/nlu/test_tail/labels.json"

with open(head_preds, "r") as f:
    head_pred_results = json.load(f)
    head_pre = [x["target"] for x in head_pred_results]
with open(tail_preds, "r") as f:
    tail_pred_results = json.load(f)
    tail_pre = [x["target"] for x in tail_pred_results]
with open(head_labels, "r") as f:
    head_gold_results = json.load(f)
    head_gold = [x["target"] for x in head_gold_results]
with open(tail_labels, "r") as f:
    tail_gold_results = json.load(f)
    tail_gold = [x["target"] for x in tail_gold_results]

assert head_gold == tail_gold
fact_gold = np.array(head_gold)
fact_pre = np.array([head_pre[i] and tail_pre[i] for i in range(len(fact_gold))])
accuracy = np.sum(fact_pre == fact_gold) / len(fact_gold)
precision = precision_score(fact_pre, fact_gold)
recall = recall_score(fact_pre, fact_gold)
f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))
result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
print(result)
