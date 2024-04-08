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
  
## Benchmarks

### SOTAB-27 and SOTAB-91

The SOTAB-27 and SOTAB-91 benchmarks were constructed from the SOTAB dataset. The SOTAB train, validation and test files, as well as instructions for their use, can be acquired via [their website.](http://webdatacommons.org/structureddata/sotab/)

### D4-20

The D4 data originated [here](https://zenodo.org/record/3647613), but our codebase includes all the necessary metadata to reconstruct the benchmark -- you can run D4Tables without any additional downloads.

### Amstr-56

The Amstr-56 evaluation CSVs can be downloaded from [here](https://drive.google.com/drive/folders/1sDWIk7ld5YaC-n9kFFCXmK8TS2FBJCd8?usp=sharing).

### PubChem-20

The PubChem-20 evaluation CSVs can be downloaded from [here](https://drive.google.com/drive/folders/1sDWIk7ld5YaC-n9kFFCXmK8TS2FBJCd8?usp=sharing).

## Training

ArcheType training makes use of the [Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca) repository and model. In order to train your own ArcheType model as described in the paper, you will need to acquire a copy of the [Alpaca-7B weights](https://huggingface.co/chavinlo/alpaca-native) as well as the training code.

### Dataset Creation

You can use our [pre-formatted training JSON](https://drive.google.com/drive/folders/1WCnPvgwkHZxH6709rAa_Ox7CgxxLvjEg?usp=share_link) for SOTAB.

You can also reproduce this JSON yourself (or generate a training JSON for another dataset of your choosing) via a two-stage process -- 

1. To generate the formatted columns, run a standard ArcheType task for the downstream architecture you intend to use, but drop the `--response` flag.
```console
python archetype/src/run.py --model_name="ArcheType-llama" --model_path=%%GEN_MODEL_PATH%% --save_path=%%JSON_OUT_PATH%% --input_files="/scratch/bf996/datasets/sotab/Train" --input_labels="/scratch/bf996/datasets/sotab/CTA_training_gt.csv" --label_set="SOTAB-91" --method ans_contains_gt gt_contains_ans resample
```
2. To convert the generated JSON to the format expected by Alpaca, you can either write a script yourself or customize ours --
```console
python archetype/src/make_dataset.py --in_file %%JSON_OUT_PATH%%
```

### Training Command

We used the following command to train our models --

```console
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=<MASTER_ADDR>:<MASTER_PORT> train.py --model_name_or_path <ALPACA_WEIGHTS> --data_path <TRAINING_JSON> --bf16 True --output_dir <SAVE_DIR --num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True
```

### Usage Examples

To reproduce our zero-shot SOTAB results using the flan-t5 architecture and S prompt, use the following command --

```console
python src/run.py --model_name="flan-t5-xxl-zs-shortprompt" --save_path=<SAVE_PATH> --input_files="<PATH>/Test" --input_labels="<PATH>/CTA_test_gt.csv" --label_set="SOTAB-91" --method ans_contains_gt gt_contains_ans resample --results --rules --response;
```

To reproduce our fine-tuned model experiments, you must also provide the path to model weights.

```console
--model_name="ArcheType-llama" --model_path=<WEIGHTS_PATH>
```

To reproduce the D4-20 results, make the following substitutions --

```console
--input_files="D4" --input_labels="D4" --label_set="D4-ZS"
```

For Amstr-56 --

```console
--input_files="<PATH>/amstr_csv" --input_labels="amstr" --label_set="amstr-ZS"
```

For PubChem-20 --

```console
--input_files="<PATH>/pubchem_csv" --input_labels="pubchem" --label_set="pubchem-ZS"
```

To reproduce the C-Baseline on T5 --

```
--model_name="flan-t5-xxl-zs-chorusprompt" --method similarity simple_random_sampling
```

To reproduce the K-Baseline on T5 --

```
--model_name="flan-t5-xxl-zs-koriniprompt" --method first_sampling
```

To vary the choice of prompt, substitute the correct prompt name in the model field.

Here are the commands for all six prompts listed in our paper (K, C, S, N, I, O)

```
--model_name="flan-t5-xxl-zs-koriniprompt" --model_name="flan-t5-xxl-zs-chorusprompt" --model_name="flan-t5-xxl-zs-shortprompt" --model_name="flan-t5-xxl-zs-noisyprompt" --model_name="flan-t5-xxl-zs-invertedprompt" --model_name="flan-t5-xxl-zs"
```

The model_name field also controls which model is called at the model querying stage.

```console
--model_name="gpt-3.5-turbo", --model_name="flan-ul2-zs", --model_name="flan-t5-xxl-zs"
```

To only evaluate a subset of the data, use the `--stop_early` flag. To reproduce the ArcheType+ results, include the `--rules` flag.

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
python archetype/src/eval.py --input_path <FILE_PATH> --label_set <LABEL_SET_NAME> --confusion_matrix
```

You can also evaluate only certain classes in the entire test set.

```
python src/eval.py --input_path <FILE_PATH> --label_set "SOTAB-91" --ignore_classes "weight, Energy, Review, Recipe/name, openingHours, Boolean, EducationalOccupationalCredential, Action, Photograph, URL, ItemList, EventAttendanceModeEnumeration, EventStatusType, DayOfWeek, ItemAvailability, RestrictedDiet, OfferItemCondition"
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

If you find this useful, please cite our work --

```
@misc{feuer2023archetype,
      title={ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models}, 
      author={Benjamin Feuer and Yurong Liu and Chinmay Hegde and Juliana Freire},
      year={2023},
      eprint={2310.18208},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgments

Datasets in this repo include variations of the "Sato" and "TURL" datasets.

Sato refers to the dataset used in ["Sato: Contextual Semantic Type Detection in Tables." Proceedings of the VLDB Endowment Vol. 13, No.11](https://github.com/megagonlabs/sato). The dataset was generated from the [VizNet](https://github.com/mitmedialab/viznet) corpus.

URL refers to the dataset used in ["TURL: table understanding through representation learning." Proceedings of the VLDB Endowment 14.3 (2020): 307-319](https://github.com/sunlab-osu/TURL). The dataset was generated from the [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/) corpus.
