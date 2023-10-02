import json
from copy import deepcopy
import os
from baseline.models import Tokenizer

import nltk
nltk.download("punkt")

linking_data_path = "data/"

fact_linking_data_file = {"persona": "persona_atomic_final_123.json", "roc": "roc_atomic_final_328.json",
                          "movie": "moviesum_atomic_final_81.json", "mutual": "mutual_atomic_final_237.json"}
fact_linking_id_file = {"persona": {"train": "persona_atomic_did_train_90.json", "val": "persona_atomic_did_val_15.json",
                                       "test": "persona_atomic_did_test_18.json"},
                        "roc": {"train": "done_sid_train_235.json", "val": "done_sid_dev_46.json",
                                "test": "done_sid_test_47.json"},
                        "movie": {"train": "done_mid_train_58.json", "val": "done_mid_dev_11.json",
                                  "test": "done_mid_test_12.json"},
                        "mutual": {"train": "mutual_atomic_did_train_170.json", "val": "mutual_atomic_did_val_33.json",
                                   "test": "mutual_atomic_did_test_34.json"}
                        }


def main():
    for split in ["train", "val", "test", "test_fact"]:
        for window in ["nlu"]:
            for task in ["entity"]:
                log_all = []
                label_all = []
                log_all_head = []
                log_all_tail = []
                for portion in ["persona", "mutual", "roc", "movie"]:

                    # data_path = linking_data_path + portion + "/" + task + "/" + window + "/" + split + "/"
                    # if not os.path.exists(data_path):
                    #     os.makedirs(data_path)
                    # print("Preprocessing Data in: " + data_path)

                    data_file = fact_linking_data_file[portion]
                    did_file = fact_linking_id_file[portion][split] if split != "test_fact" else fact_linking_id_file[portion]["test"]

                    with open(linking_data_path + portion + "/" + data_file, 'r') as f:
                        linking_raw_data = json.load(f)
                    with open(linking_data_path + portion + "/" + did_file, 'r') as f:
                        linking_cid = json.load(f)

                    # log = []
                    # label = []

                    for cid in linking_cid:
                        max_turn = len(linking_raw_data[str(cid)]["text"])

                        for tid, fact_turn in linking_raw_data[str(cid)]["facts"].items():
                            sample = {"cid": str(cid), "tid": int(tid), "text": []}

                            ''''''
                            left = max(0, int(tid)-2)
                            if window == "nlu":
                                right = min(max_turn, int(tid)+3)
                            elif window == "nlg":
                                right = int(tid) + 1
                            else:
                                raise ValueError("window not in ['center', 'nlg']")

                            for utter in linking_raw_data[str(cid)]["text"][left:int(tid)]:
                                sample["text"].append({"type": "p_context", "utter": utter.lower()})
                            for utter in linking_raw_data[str(cid)]["text"][int(tid)+1:right]:
                                sample["text"].append({"type": "f_context", "utter": utter.lower()})

                            sample["text"].append({"type": "center",
                                                   "utter": linking_raw_data[str(cid)]["text"][int(tid)].lower()})

                            if split != "test_fact":
                                for head, triples in fact_turn.items():
                                    sample_single = deepcopy(sample)
                                    sample_single["fid"] = -1
                                    sample_single["text"].append({"type": "fact", "utter": head.lower()})
                                    # log.append(sample_single)
                                    log_all.append(sample_single)
                                    if triples["confidence"] < 0.49:
                                        # label.append({"target": False, "linking": None})
                                        label_all.append({"target": False, "linking": None})
                                    else:
                                        # label.append({"target": True, "linking": None})
                                        label_all.append({"target": True, "linking": None})
                                        for fid, rt in enumerate(triples["triples"]):
                                            relevance, relation = rt["final"], rt["relationship"]
                                            if relation in ["irr", "rpa", "rpp", "rpf"]:
                                                sample_single = deepcopy(sample)
                                                sample_single["fid"] = fid
                                                sample_single["text"].append({"type": "fact", "utter": rt["tail"].lower()})
                                                if relevance in ["always", "sometimes"]:
                                                    log_all.append(sample_single)
                                                    label_all.append({"target": True, "linking": relation})
                                                else:
                                                    log_all.append(sample_single)
                                                    label_all.append({"target": False, "linking": None})
                            else:
                                for head, triples in fact_turn.items():
                                    for fid, rt in enumerate(triples["triples"]):
                                        sample_single = deepcopy(sample)
                                        sample_single["fid"] = -1
                                        sample_single["text"].append({"type": "fact", "utter": head.lower()})
                                        log_all_head.append(sample_single)

                                        sample_single = deepcopy(sample)
                                        sample_single["fid"] = fid
                                        sample_single["text"].append({"type": "fact", "utter": rt["tail"].lower()})
                                        log_all_tail.append(sample_single)
                                        if triples["confidence"] < 0.49 or rt["final"] not in ["always", "sometimes"]:
                                            label_all.append({"target": False, "linking": None})
                                        else:
                                            label_all.append({"target": True, "linking": rt["relationship"]})

                    '''
                    if split == "train":
                        tokenizer.construct()
                        tokenizer.save_vocab(data_path)
                    with open(data_path + "logs.json", "w") as f:
                        json.dump(log, f, indent=2)
                    with open(data_path + "labels.json", "w") as f:
                        json.dump(label, f, indent=2)
                    '''

                data_path_all = linking_data_path + "all/" + task + "/" + window + "/" + split + "/"
                if not os.path.exists(data_path_all):
                    os.makedirs(data_path_all)
                # if split == "train":
                #     tokenizer_all.construct()
                #     tokenizer_all.save_vocab(data_path_all)
                if split != "test_fact":
                    data_path_all = linking_data_path + "all/" + task + "/" + window + "/" + split + "/"
                    if not os.path.exists(data_path_all):
                        os.makedirs(data_path_all)
                    with open(data_path_all + "logs.json", "w") as f:
                        json.dump(log_all, f, indent=2)
                    with open(data_path_all + "labels.json", "w") as f:
                        json.dump(label_all, f, indent=2)
                else:
                    data_path_all = linking_data_path + "all/" + task + "/" + window + "/test_head/"
                    if not os.path.exists(data_path_all):
                        os.makedirs(data_path_all)
                    with open(data_path_all + "logs.json", "w") as f:
                        json.dump(log_all_head, f, indent=2)
                    with open(data_path_all + "labels.json", "w") as f:
                        json.dump(label_all, f, indent=2)

                    data_path_all = linking_data_path + "all/" + task + "/" + window + "/test_tail/"
                    if not os.path.exists(data_path_all):
                        os.makedirs(data_path_all)
                    with open(data_path_all + "logs.json", "w") as f:
                        json.dump(log_all_tail, f, indent=2)
                    with open(data_path_all + "labels.json", "w") as f:
                        json.dump(label_all, f, indent=2)

if __name__ == "__main__":
    main()
