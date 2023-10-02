import random
import logging

from itertools import chain
from copy import deepcopy

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences, truncate_sequences_dual
)

from .utils.dataset_walker import DatasetWalker

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<c_sep>", "<f_sep>", "<p_b>", "<u_b>", "<f_b>"]
}
SPECIAL_TOKENS_VALUES = ["<c_sep>", "<f_sep>", "<p_b>", "<u_b>", "<f_b>"]
ADD_TOKENS_VALUES = ["[FSEP]"]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.task_type = args.task

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.context_sep = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[0])
        self.fact_sep = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[1])
        self.past_start = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[2])
        self.utter_start = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[3])
        self.future_start = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS_VALUES[4])
        self.pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)

        self.samples = self._prepare_samples()
        self._create_examples()

    def _prepare_samples(self):
        logger.info("Prepare fact generation samples")
        samples = []
        # only show progress bar in one process
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])):
            sample = {}
            if log.get("cid") is not None:
                sample["cid"] = log["cid"]
            else:
                sample["cid"] = -1
            if log.get("tid") is not None:
                sample["tid"] = log["tid"]
            else:
                sample["tid"] = -1
            if log.get("hid") is not None:
                sample["hid"] = log["hid"]
            else:
                sample["hid"] = -1
            if log.get("fid") is not None:
                sample["fid"] = log["fid"]
            else:
                sample["fid"] = -1
            sample["text"] = log["text"]
            sample["label"] = label
            samples.append(deepcopy(sample))
        return samples

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for sample in tqdm(self.samples, disable=self.args.local_rank not in [-1, 0]):
            context_id = sample["cid"]
            turn_id = sample["tid"]
            head_id = sample["hid"]
            fact_id = sample["fid"]
            label = sample["label"]
            text = sample["text"]

            if self.task_type == "generation":
                if label is None:
                    label = {"target": ""}
                target = None
                lm_target = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label["target"]))
            else:
                if label is None:
                    label = {"target": False, "linking": None}
                target = label["target"]
                lm_target = None

            p_context = []
            f_context = []
            fact = []
            utterance = []
            for line in text:
                if line["type"] == "p_context":
                    p_context.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"])))
                if line["type"] == "f_context":
                    f_context.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"])))
                if line["type"] == "fact":
                    fact.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"])))
                if line["type"] == "center":
                    utterance = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line["utter"]))

            # perform token-level truncation of history from the left
            # <s>, </s></s>, </s>, "<c_sep>", "<f_sep>", "<p_b>", "<u_b>", "<f_b>"
            no_trunc_token_num = 2 + len(p_context) + len(f_context) + len(fact) + len(utterance)
            # no_trunc_token_num = 2 + len(fact) + len(utterance)
            for node in fact:
                no_trunc_token_num += len(node)  # do not truncate statement

            truncated_p_context = truncate_sequences(deepcopy(p_context),
                                                     (self.args.max_tokens - no_trunc_token_num) // 2)
            truncated_f_context = truncate_sequences(deepcopy(f_context),
                                                     (self.args.max_tokens - no_trunc_token_num) // 2)

            self.examples.append({
                "p_context": truncated_p_context,
                "f_context": truncated_f_context,
                "utterance": utterance,
                "fact": fact,
                "label": label,
                "target": target,
                "lm_target": lm_target,
                "context_id": context_id,
                "turn_id": turn_id,
                "head_id": head_id,
                "fact_id": fact_id
            })

    def build_input_from_segments(self, p_context, f_context, utterance, fact, lm_target=None):
        """ Build a sequence of input from example """
        instance = {}
        sequence_context = []

        if len(p_context) > 0:
            sequence_context.append(self.past_start)
            sequence_context += deepcopy(p_context[0])
            for utter in p_context[1:]:
                sequence_context.append(self.context_sep)
                sequence_context += deepcopy(utter)

        sequence_context.append(self.utter_start)
        sequence_context += deepcopy(utterance)

        if len(f_context) > 0:
            sequence_context.append(self.future_start)
            sequence_context += deepcopy(f_context[0])
            for utter in f_context[1:]:
                sequence_context.append(self.context_sep)
                sequence_context += deepcopy(utter)

        if len(fact) > 0:
            sequence_fact = deepcopy(fact[0])
            for node in deepcopy(fact[1:]):
                sequence_fact.append(self.fact_sep)
                sequence_fact += deepcopy(node)
            sequence = [sequence_context, sequence_fact]
            instance["input_ids"] = self.tokenizer.build_inputs_with_special_tokens(sequence_context, sequence_fact)
        else:
            sequence = sequence_context
            instance["input_ids"] = self.tokenizer.build_inputs_with_special_tokens(sequence_context)

        instance["token_type_ids"] = None
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        if self.task_type == "generation":
            instance["lm_target_ids"] = self.tokenizer.build_inputs_with_special_tokens(deepcopy(lm_target))
        else:
            instance["lm_target_ids"] = None

        return instance, sequence

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)

    '''
    def build_input_from_segments(self, utterance, fact, lm_target=None):
        """ Build a sequence of input from example """
        instance = {}
        sequence_context = []

        sequence_context.append(self.utter_start)
        sequence_context += deepcopy(utterance)

        if len(fact) > 0:
            sequence_fact = deepcopy(fact[0])
            # for node in deepcopy(fact[1:]):
            #     sequence_fact.append(self.fact_sep)
            #     sequence_fact += deepcopy(node)
            sequence = [sequence_context, sequence_fact]
            instance["input_ids"] = self.tokenizer.build_inputs_with_special_tokens(sequence_context, sequence_fact)
        else:
            sequence = sequence_context
            instance["input_ids"] = self.tokenizer.build_inputs_with_special_tokens(sequence_context)

        instance["token_type_ids"] = None
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        if self.task_type == "generation":
            instance["lm_target_ids"] = self.tokenizer.build_inputs_with_special_tokens(deepcopy(lm_target))
        else:
            instance["lm_target_ids"] = None

        return instance, sequence

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)
    '''


class FactLinkingDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactLinkingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance, _ = self.build_input_from_segments(example["p_context"], example["f_context"], example["utterance"],
                                                     example["fact"])
        # instance, _ = self.build_input_from_segments(example["utterance"], example["fact"])
        instance["label"] = example["target"]
        instance["context_id"] = example["context_id"]
        instance["turn_id"] = example["turn_id"]
        instance["head_id"] = example["head_id"]
        instance["fact_id"] = example["fact_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        # token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "context_ids": [ins["context_id"] for ins in batch],
            "turn_ids": [ins["turn_id"] for ins in batch],
            "head_ids": [ins["head_id"] for ins in batch],
            "fact_ids": [ins["fact_id"] for ins in batch]
        }

        # pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, self.pad_token))
        token_type_ids = torch.full_like(input_ids, 0)
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, 0)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info


class FactGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance, _ = self.build_input_from_segments(example["p_context"], example["f_context"], example["utterance"],
                                                     example["fact"], example["lm_target"])
        # instance, _ = self.build_input_from_segments(example["utterance"], example["fact"], example["lm_target"])

        instance["context_id"] = example["context_id"]
        instance["turn_id"] = example["turn_id"]
        instance["head_id"] = example["head_id"]
        instance["fact_id"] = example["fact_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_target_ids = [ins["lm_target_ids"] for ins in batch]

        data_info = {
            "context_ids": [ins["context_id"] for ins in batch],
            "turn_ids": [ins["turn_id"] for ins in batch],
            "head_ids": [ins["head_id"] for ins in batch],
            "fact_ids": [ins["fact_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad_token))

        token_type_ids = torch.full_like(input_ids, 0)

        decoder_input_ids = torch.tensor(pad_ids(lm_target_ids, self.pad_token))
        decoder_input_ids = decoder_input_ids[:, :-1].contiguous()

        lm_label_ids = torch.tensor(pad_ids(lm_target_ids, -100))
        lm_label_ids = lm_label_ids[:, 1:].contiguous()

        return input_ids, token_type_ids, decoder_input_ids, lm_label_ids, data_info


class FactGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance, _ = self.build_input_from_segments(example["p_context"], example["f_context"], example["utterance"],
                                                     example["fact"], example["lm_target"])
        # instance, _ = self.build_input_from_segments(example["utterance"], example["fact"], example["lm_target"])
        instance["lm_target_text"] = example["label"]["target"]
        instance["context_id"] = example["context_id"]
        instance["turn_id"] = example["turn_id"]
        instance["head_id"] = example["head_id"]
        instance["fact_id"] = example["fact_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_target_text = [ins["lm_target_text"] for ins in batch]

        data_info = {
            "context_ids": [ins["context_id"] for ins in batch],
            "turn_ids": [ins["turn_id"] for ins in batch],
            "head_ids": [ins["head_id"] for ins in batch],
            "fact_ids": [ins["fact_id"] for ins in batch]
        }

        # pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, self.pad_token))

        token_type_ids = torch.full_like(input_ids, 0)

        return input_ids, token_type_ids, lm_target_text, data_info
