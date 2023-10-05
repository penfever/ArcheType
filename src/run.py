import os, json, requests
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import argparse
import openai

try:
  from .model import init_model, get_sent_model, get_model_resp, get_sherlock_resp, get_coherence_scores, seed_all, free_memory
  from .data import get_df_sample, fix_labels, insert_source, get_lsd, get_d4_dfs, pd_read_any
  from .metrics import results_checker, results_checker_doduo
  from .const import DOTENV_PATH, MAX_LEN
except ImportError:
  from model import init_model, get_sent_model, get_model_resp, get_sherlock_resp, get_coherence_scores, seed_all, free_memory
  from data import get_df_sample, fix_labels, insert_source, get_lsd, get_d4_dfs, pd_read_any
  from metrics import results_checker, results_checker_doduo
  from const import DOTENV_PATH, MAX_LEN


def run(
    model_name : str, 
    save_path : str, 
    inputs : list, 
    label_set : list, 
    input_df : pd.DataFrame, 
    resume : bool = True, 
    results : bool = True, 
    stop_early : int = -1, 
    rand_seed : int = 13, 
    sample_size : int = 5, 
    link : str = None, 
    response : bool = True, 
    summ_stats : bool = False, 
    table_src : bool = False, 
    other_col : bool = False, 
    skip_short : bool = False, 
    min_var : int = 0, 
    method : list = ["similarity"],
    args : dict = dict()):
  
  seed_all(rand_seed)
  args['context_is_numeric'] = False

  if model_name in ["llama", "llama-old", "sherlock"]:
    pass
  elif "gpt-3.5" in model_name:
    load_dotenv(DOTENV_PATH)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    assert openai.api_key != None, "api key did not load"
  else:
    init_model(model_name, args)
  infmods = "sherlock" in model_name or "doduo" in model_name
  isd4 = "d4" in label_set['name']
  if "similarity" in method:
    get_sent_model(args)
  if resume and os.path.isfile(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
      prompt_dict = json.load(f)
  else:
    prompt_dict = {}
  s = requests.Session()
  if "-zs" in model_name:
    args["base_model"].eval()
  if isinstance(inputs, dict):
    labels = ["_".join(k.split("_")[:-1]) for k in inputs.keys()]
    inputs = list(inputs.values())
  for idx, f in tqdm(enumerate(inputs), total=len(inputs)):
    # try:
    #     free_memory()
    # except Exception as e:
    #     print(f"Failed to free memory, error message was: \n {e}")
    if idx % 10 == 0:
      with open(save_path, 'w', encoding='utf-8') as alt_f:
        json.dump(prompt_dict, alt_f, ensure_ascii=False, indent=4)
    if stop_early > -1 and idx == stop_early:
      break
    if isd4:
        f_df = f
        label_indices=[2]
        gt_labels = labels[idx]
    elif "skip-eval" in method:
        f_df = pd_read_any(f)
        gt_labels = None
        label_indices = [i for i in range(len(f_df.columns))]
    else:
        gt_labels = input_df[input_df['table_name'] == f.name]
        label_indices = pd.unique(gt_labels['column_index']).tolist()
        f_df = pd.read_json(f, compression='gzip', lines=True)
    if infmods:
        label_indices = pd.unique(gt_labels['column_index']).tolist()
        # label_indices = ["values"]
        key = get_sherlock_resp(f_df, gt_labels, prompt_dict, model_name, label_indices, str(f), label_set, args)
        continue
    if "coherence_sampling" in method:
        coherence_scores = get_coherence_scores(f_df, model_name, args)
    else:
        coherence_scores = None
    sample_df = get_df_sample(f_df, rand_seed, label_indices, sample_size, full=summ_stats, other_col=other_col, max_len=args["MAX_LEN"], method=method, coherence_scores=coherence_scores, args=args)
    f_df_cols = f_df.columns
    for idx, col in enumerate(f_df_cols):
      if idx not in label_indices:
        continue
      #NOTE: skipping evaluation for columns with insufficient variance in the column
      if len(pd.unique(sample_df.astype(str)[col])) < min_var:
        continue
      if isd4:
        orig_label = gt_labels
      elif "skip-eval" in method:
        orig_label = ""
      else:
        gt_row = gt_labels[gt_labels['column_index'] == idx]
        orig_label = gt_row['label'].item()
      label = fix_labels(orig_label, label_set)
      limited_context = sample_df[col].tolist()[:sample_size]
      #NOTE: could consider using min_var here
      #if full and len(pd.unique(sample_df[col].tolist())) < 3:
      if table_src:
        if "zs" in model_name:
          context_n = insert_source(sample_df[col].tolist(), f.name, zs="zs" in model_name)
          args["table_name"] = context_n
          context = sample_df[col].tolist()
        else:
          context = insert_source(sample_df[col].tolist(), f.name, zs="zs" in model_name)
      else:
        context = sample_df[col].tolist()      
      key = get_model_resp(label_set, context, label, prompt_dict, link=link, response=response, session=s, cbc=None, model=model_name, limited_context=limited_context, method=method, args=args)
      prompt_dict[key]['original_label'] = orig_label
      prompt_dict[key]['file+idx'] = str(f) + "_" + str(idx)
  with open(save_path, 'w', encoding='utf-8') as my_f:
    json.dump(prompt_dict, my_f, ensure_ascii=False, indent=4)
  if results:
    if "skip-eval" in method:
      print("Cannot evaluate, no ground truth labels provided.")
      return
    print("Run complete. Checking results...")
    if model_name == "doduo":
      results_checker_doduo(save_path)
    else:
      results_checker(save_path, skip_duplicates = False, naive_score = False)

def main():
    parser = argparse.ArgumentParser(description="Takes input parameters for the 'run' function.")
    
    parser.add_argument("--model_name", type=str, help="Model name: supported models and their names are listed in init_model", required=True)
    parser.add_argument("--model_path", type=str, help="Path to ArcheType-LLAMA or zs-LLAMA model weights", default="")
    parser.add_argument("--save_path", type=str, help="Save path", required=True)
    parser.add_argument("--input_files", type=str, help="Path to input CSV files", required=True)
    parser.add_argument("--label_set", type=str, help="Name of label set (SOTAB-91, SOTAB-55, SOTAB-27, D4-ZS, D4-DoDuo, custom)", required=True)
    parser.add_argument("--custom-labels", nargs='+', type=str, help="Custom labels", required=False)
    parser.add_argument("--input_labels", type=str, help="Path to input DataFrame (CSV file) for SOTAB. skip-eval will generate predictions but will not compare them to anything. D4 will use (internal) D4 ground-truth labels.", required=True)
    parser.add_argument("--resume", action='store_true', help="Resume")
    parser.add_argument("--results", action='store_true', help="Results")
    parser.add_argument("--stop_early", type=int, default=-1, help="Stop early")
    parser.add_argument("--rand_seed", type=int, default=1902582, help="Random seed")
    parser.add_argument("--sample_size", type=int, default=5, help="Sample size")
    parser.add_argument("--link", type=str, help="Link")
    parser.add_argument("--response", action='store_true', help="If this flag is not called, the model will not provide a prediction. No response is useful for generating training datasets or testing workflows.")
    parser.add_argument("--summ_stats", action='store_true', help="Summary statistics")
    parser.add_argument("--table_src", action='store_true', help="Table source")
    parser.add_argument("--other_col", action='store_true', help="Other column")
    parser.add_argument("--skip_short", action='store_true', help="Skip short")
    parser.add_argument("--min_var", type=int, default=0, help="Columns which contain less than min_var unique values will not be evaluated")
    parser.add_argument("--method", nargs='+', type=str, default=["similarity"], help="Sets label remapping strategy. skip-existing will skip columns which already have a prompt in the prompt_dict. If skip-eval is in method, then no evaluation will be performed. similarity will use the similarity metric to find the closest label in the label set. ans_contains_gt and gt_contains_ans will use contains label remapping, and resample will call the LLM multiple times. If check_labels is in method, then every ground truth label will be verified against the values in the label set.")

    args = parser.parse_args()

    if args.input_files == "D4":
      input_files = get_d4_dfs()
    else:
      # Define the file extensions to search for
      extensions = ('.json', '.csv', '.json.gz', '.parquet', '.xlsx', '.xls', '.tsv')
      input_files = []
      for extension in extensions:
        input_files = input_files + list(Path(args.input_files).rglob(f"**/*{extension}"))
    
    if args.input_labels == "D4":
      input_df = None
    elif args.input_labels == "skip-eval":
      input_df = None
      args.method = args.method + ["skip-eval"]
    else:
      input_df = pd.read_csv(args.input_labels)

    if args.label_set == "custom":
      label_set = {"name" : "custom", "label_set" : args.custom_labels, "dict_map" : {c : c for c in args.custom_labels}, 'abbrev_map' : {c : c for c in args.custom_labels}}
    else:
      label_set = get_lsd(args.label_set)

    arg_dict = {"MAX_LEN" : MAX_LEN, "model_path" : args.model_path}

    run(
        args.model_name,
        args.save_path,
        input_files,
        label_set,
        input_df,
        args.resume,
        args.results,
        args.stop_early,
        args.rand_seed,
        args.sample_size,
        args.link,
        args.response,
        args.summ_stats,
        args.table_src,
        args.other_col,
        args.skip_short,
        args.min_var,
        args.method,
        arg_dict
    )

if __name__ == "__main__":
    main()