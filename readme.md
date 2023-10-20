# ArcheType: A novel framework for open-source column type annotation using large language models 

This repository contains the codebase of our paper ArcheType: A novel framework for open-source column type annotation using large language models, available at arXiv.

## Hardware Requirements

ArcheType requires an NVIDIA GPU with at least 12GB VRAM.

## Installation

```console
$ git clone [link to repo]
$ cd archetype
$ pip install -r requirements.txt 
```

With `conda`, create a virtual environment and install the required packages as below:


```console
$ conda create --name archetype python=3.7.10
$ conda activate archetype
$ pip install -r requirements.txt
```

### Doduo

Doduo must be installed separately -- please follow the instructions in [this repository](https://github.com/megagonlabs/doduo).

Don't forget to add Doduo to your Python path.

```console
$export PYTHONPATH=<DODUO_PATH>:$PYTHONPATH
```

### Editing Paths

In const.py, replace the ARCHETYPE_PATH variable with the absolute path to your ArcheType installation.
In const.py, replace the DOTENV_PATH variable with the absolute path to the directory containing your dotenv with your OpenAI API key.
  
## Data Preparation

### SOTAB

The SOTAB train, validation and test files, as well as instructions for their use, can be acquired via [their website.](http://webdatacommons.org/structureddata/sotab/)

### D4 Tables

The D4 dataset can be acquired [here](https://zenodo.org/record/3647613).

## Training

ArcheType training makes use of the [Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca) repository and model. In order to train your own ArcheType model as described in the paper, you will need to acquire a copy of the [Alpaca-7B weights](https://huggingface.co/chavinlo/alpaca-native) as well as the training code.

You can use our [pre-formatted training JSON](https://drive.google.com/drive/folders/1WCnPvgwkHZxH6709rAa_Ox7CgxxLvjEg?usp=share_link) for SOTAB.

We used the following command to train our models --

```console
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=<MASTER_ADDR>:<MASTER_PORT> train.py --model_name_or_path <ALPACA_WEIGHTS> --data_path <TRAINING_JSON> --bf16 True --output_dir <SAVE_DIR --num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True
```
  
## Inference

We provide a command line interface to reproduce the experimental results described in the paper.

The inference code will produce a JSON file at the output path you specify using --save_path containing model responses, ground-truth responses and metadata.

### Usage Examples

To reproduce the ArcheType-T5 experiment on SOTAB-27, use the following command;

```console
$ python archetype/src/run.py --model_name="flan-t5-xxl-zs" --save_path="<SAVE_PATH>/results.json" --input_files="<SOTAB_PATH>/Test" --input_labels="<SOTAB_PATH>/CTA_test_gt.csv" --label_set="SOTAB-27" --method ans_contains_gt gt_contains_ans resample --results --response
```

To reproduce our fine-tuned model experiments, you must also provide the path to model weights and change the label set.

```console
--model_name="ArcheType-llama" --model_path=<WEIGHTS_PATH> --label_set="SOTAB-91"
```

To reproduce the D4Tables results:

```console
$ python archetype/src/run.py --model_name="flan-t5-xxl-zs" --save_path="<SAVE_PATH>/results.json" --input_files="D4" --input_labels="D4" --label_set="D4-ZS" --method ans_contains_gt gt_contains_ans resample --results --response
```

To reproduce the DoDuo results on SOTAB-27:

```console
$ python archetype/src/run.py --model_name="doduo" --save_path="<SAVE_PATH>/results.json" --input_files="<SOTAB_PATH>/Test" --input_labels="<SOTAB_PATH>/CTA_test_gt.csv" --label_set="SOTAB-27" --results --response
```

To reproduce the OpenAI GPT-3.5 experiment on SOTAB-91;

```console
$ python archetype/src/run.py --model_name="gpt-3.5" --save_path="<SAVE_PATH>/results.json" --input_files="<SOTAB_PATH>/Test" --input_labels="<SOTAB_PATH>/CTA_test_gt.csv" --label_set="SOTAB-91" --method ans_contains_gt gt_contains_ans resample --results --response
```

## Custom Labels and Data

Any LLM supported by ArcheType can be used to perform zero-shot CTA on your own custom data from the command line, using a label set of your choice.

Your data can be in any format recognized by Pandas, including csv, tsv, Excel, parquet and sql.

This example uses ArcheType-gpt-3.5, and has a set of four custom labels: text number id, and place.

```console
$ python archetype/src/run.py --model_name="gpt-3.5" --save_path="<SAVE_PATH>/results.json" --input_files="<CUSTOM_PATH>/Test" --input_labels="skip-eval" --label_set="custom" --custom-labels text number id place --method ans_contains_gt gt_contains_ans resample --response
```

## Evaluation

You can evaluate a results json using eval.py. Add --confusion_matrix to generate confusion matrices at the same time.

```
python archetype/src/eval.py --input_path "results/flan-ul2-zs-shortprompt-pubchem-mod.json" --label_set "pubchem-ZS" --confusion_matrix
```

## Label Sets in our Paper

SOTAB-27: 
```
[
 'Boolean',
 'Coordinates',
 'Country',
 'CreativeWork',
 'Date',
 'Event',
 'Gender',
 'JobPosting',
 'Language',
 'Company',
 'Number',
 'Organization',
 'Person',
 'Product',
 'SportsTeam',
 'Text',
 'Time',
 'URL',
 'category',
 'currency',
 'email',
 'price',
 'streetAddress',
 'telephone',
 'Age',
 'weight',
 'zipCode']
```

SOTAB-91: See the [associated site.](http://webdatacommons.org/structureddata/sotab/#toc8)

D4Tables:
```
['School ID',
 'Ethnicity',
 'Letter Grade',
 'Educational Organization',
 'School DBN',
 'Region in Brooklyn',
 'Region in Bronx',
 'Permit Type',
 'Region in Queens',
 'Region in Manhattan',
 'Region in Staten Island',
 'County',
 'Elevator or Staircase',
 'Short City Agency Name',
 'Color',
 'Full City Agency Name',
 'Country',
 'State',
 'Month',
 'License plate type']
```

PubChemTables:
```
['Person\'s First Name and Middle Initials', 
'Molecular Formula', 
'Book Title', 
'Cell Alternative Label', 
'Book Title',
'Disease Alternative Label', 
'MD5 Hash', 
'Person\'s Last Name', 
'Biological Formula', 
'Taxonomy Label',
'InChI (International Chemical Identifier)', 
'SMILES (Simplified Molecular Input Line Entry System)', 
'Abstract for Patent', 
'Organization', 
'Book ISBN', 
'Concept Broader Term', 
'Journal Title', 
'Chemical',
'Person\'s Full Name', 
'Journal ISSN', 
'Patent Title']
```

AmstrTables

Here, `<STATE_NAME>` stands in for all fifty U.S. states.

```
['Newspaper or Publication',
'Numeric Identifier',
'Town',
'State',
'Headline',
'Author Byline',
'Article from <STATE_NAME>]
```

## Citation

```
TBD
```

## Acknowledgments

Datasets in this repo include variations of the "Sato" and "TURL" datasets.

Sato refers to the dataset used in ["Sato: Contextual Semantic Type Detection in Tables." Proceedings of the VLDB Endowment Vol. 13, No.11](https://github.com/megagonlabs/sato). The dataset was generated from the [VizNet](https://github.com/mitmedialab/viznet) corpus.

URL refers to the dataset used in ["TURL: table understanding through representation learning." Proceedings of the VLDB Endowment 14.3 (2020): 307-319](https://github.com/sunlab-osu/TURL). The dataset was generated from the [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/) corpus.
