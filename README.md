# ComFact-Relation-Agnostic
This is the repository for relation-agnostic (entity-based) fact linking on [ComFact](https://arxiv.org/abs/2210.12678) benckmark.

## Getting Started
Start with creating a **python 3.6** venv and installing **requirements.txt**.

## ComFact Datasets
The **ComFact dataset** can be downloaded from [this link](https://drive.google.com/file/d/1nbQiASv32WTGVo5TQHatJbxBlz2HtMRP/view?usp=sharing), please place data/ under this root directory.

## Data Preprocessing
```
python process_nlu_head_tail_link.py
```

## Relation-Agnostic Entity Linker Training
```
bash train_baseline_entity_linker.sh
```
Our trained entity linker based on DeBERTa (large) can be downloaded from [this link](https://drive.google.com/drive/folders/1204HllA462K6FeBO3pMH8v1G-i8y16Iz?usp=sharing), please place all-deberta-large-nlu-entity/ under the runs/ folder.

## Fact Linking Evaluation
```
bash run_baseline_head_entity_link.sh
bash run_baseline_tail_entity_link.sh
python merge_linking.py
```
