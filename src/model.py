from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from sentence_transformers import SentenceTransformer
import torch
import openai
import requests
from retry import retry

import os
import argparse

try:
    from .const import *
    from .data import *
    from .match import *
except ImportError:
    from const import *
    from data import *
    from match import *

def model_coherence_call(s, model_name, args):
    if not s or s == "" or not isinstance(s, str):
        return 1
    prompt = f'On an integer scale of 1 to 10, please score how coherent the following English text is. TEXT: {s[:256]} \n'
    res = query_correct_model(model_name, prompt, "", "", None, "", None, args=args)
    if res in range(1,11):
        return res
    return 1

def get_coherence_scores(f_df, model_name, args):
    coherence_scores = []
    for col in f_df.columns:
        colvals = f_df[col]
        if all(colvals.astype(str).apply(str.isnumeric)):
            coherence_scores.append(pd.Series([1 for i in range(len(colvals))]))
        else:
            coherence_scores.append(pd.Series([model_coherence_call(s, model_name, args) for s in colvals.tolist()]))
    return coherence_scores

def query_correct_model(model, prompt, context_labels, context, session, link, lsd, args):
    if "gpt" in model:
        orig_ans = call_gpt_model(prompt, lsd)
    elif any(["llama-zs" in model, "opt-iml-max-30b-zs" in model, "ArcheType-llama" in model, "ArcheType-llama-oc" in model]):
        try:
            orig_ans = args["llm_chain"].run(prompt)
        except NameError:
            set_pipeline(k=1, args=args)
            orig_ans = args["llm_chain"].run(prompt)
    elif any(["topp-zs" in model, "flan-t5-xxl-zs" in model, "flan-ul2-zs" in model]):
        orig_ans = get_topp_resp(prompt, 1, args)
    else:
        orig_ans = call_llama_model(session, link, prompt, lsd, None, args)
    # print("Original answer: ", orig_ans)
    return orig_ans

def call_llama_model(session, link, prompt, lsd, var_params, args):
    if session:
      ans = session.post(link, json=make_json(prompt, var_params, args))
    else:
      ans = requests.post(link, json=make_json(prompt, var_params, args))
    ans = ans.json()["data"]
    ans_n = fix_labels(ans[0][len(prompt):].strip(), lsd)
    return ans_n

def call_gpt_model(prompt, lsd):
    ans = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "user", "content": prompt},
      ],
      temperature=0,
    ).choices[0]['message']['content']
    ans_n = fix_labels(ans, lsd)
    return ans_n

def get_topp_resp(prompt, k, args):
    inputs = args["tokenizer"].encode(prompt, return_tensors="pt").cuda()
    outputs = args["base_model"].generate(inputs, 
                                  max_length=args["MAX_LEN"],
                                  temperature=0.1*k,
                                  top_p=0.90-(0.1 * k),
                                  do_sample=True,
                                  repetition_penalty=1.3
                                  )
    orig_ans = args["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    return orig_ans

@retry(Exception, tries=3, delay=3)
def get_model_resp(lsd: dict, context : list, ground_truth : str, prompt_dict : dict, link : str, response = True, session=None, cbc=None, model="llama", limited_context=None, method = ["ans_contains_gt", "gt_contains_ans", "resample"], args = dict()):
  isd4 = "d4" in lsd['name']
  if isd4:
      gtv = lsd['d4_map'][ground_truth]
      if isinstance(gtv, str):
        gtv = [gtv]
      ground_truth = [s.lower() for s in gtv]
  if "hierarchical" in method and not isd4:
      dtype = get_base_dtype(limited_context)
      fixed_labels = sotab_top_hier[dtype]
  else:
      fixed_labels = list(set([fix_labels(s, lsd) for s in lsd['label_set']]))
      ground_truth = fix_labels(ground_truth, lsd)
  if "check_labels" in method:
    assert ground_truth in fixed_labels, f"Ground truth {ground_truth} not in label set {fixed_labels}"
  context_labels = ", ".join(fixed_labels)
  fixed_labels = sorted(fixed_labels, key=len, reverse=True)
  if model in ["llama-zs", "opt-iml-30b-zs", "ArcheType-llama", "ArcheType-llama-oc"]:
    set_pipeline(k=1, args=args)
  prompt = prompt_context_insert(context_labels, context, args["MAX_LEN"], model)
  d_p = prompt_dict.get(prompt, -1)
  #skip existing logic
  if d_p != -1 and "skip-existing" in method:
    # recompute_results(prompt_dict, prompt, "llama", cbc, lsd)
    return prompt
  elif d_p != -1:
    while prompt_dict.get(prompt, -1) != -1:
        prompt = prompt + "*"
  #response logic
  if not response:
    orig_ans = ans_n = ""
  else:
    orig_ans = apply_basic_rules(limited_context, None, lsd)
    if orig_ans is None:
        orig_ans = query_correct_model(model, prompt, context_labels, context, session, link, lsd, args)
        #hierarchical matching logic
        if "hierarchical" in method and dtype == "other" and orig_ans not in ['email', 'URL', 'WebHTMLAction', 'Photograph']:
            next_label_set = sotab_other_hier.get(orig_ans, -1)
            if next_label_set == -1:
                print(f"Original answer {orig_ans} not found in hierarchy")
                next_label_set = sotab_other_hier['text']
            fixed_labels = list(set([fix_labels(s, lsd) for s in next_label_set])) 
            context_labels = ", ".join(fixed_labels)
            fixed_labels = sorted(fixed_labels, key=len, reverse=True)
            orig_ans = query_correct_model(model, prompt, context_labels, context, session, link, lsd, args)  
        #fuzzy matching logic
        ans_n = fuzzy_label_match(orig_ans, fixed_labels, session, link, prompt, lsd, model, method=method, args=args).lower()
    else:
        ans_n = orig_ans.lower()
  if isd4:
    res = ans_n in ground_truth
  elif "skip-eval" in method:
    res = False
  else:
    res = ans_n == ground_truth
  ans_dict = {"response" : ans_n, "context" : context, "ground_truth" : ground_truth, "correct" : res, "original_model_answer" : orig_ans}
  prompt_dict[prompt] = ans_dict
  return prompt

def get_sent_model(args):
    args["sent_model"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return

def set_pipeline(k=1, args=None):
    args["pipe"] = pipeline(
        "text-generation",
        model=args["base_model"], 
        tokenizer=args["tokenizer"], 
        max_length=args["MAX_LEN"],
        temperature=0.5*k,
        top_p=0.80-(0.1 * k),
        do_sample=True,
        repetition_penalty=1.3
    )
    args["local_llm"] = HuggingFacePipeline(pipeline=args["pipe"])
    args["llm_chain"] = LLMChain(
        prompt=args["pt"], 
        llm=args["local_llm"]
    )
    return args

def init_model(model, args):
    if model == "doduo":
        from doduo.doduo import Doduo
    with torch.no_grad():
        torch.cuda.empty_cache()
    if "llama" in model: 
        LLAMA_PATH = args["model_path"]
        args["MAX_LEN"]=2048
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)
        config = AutoConfig.from_pretrained(LLAMA_PATH,
                                            torch_dtype=torch.float16,
                                            load_in_8bit=True)
        with init_empty_weights():
            base_model = AutoModelForCausalLM.from_config(config)
        base_model.tie_weights()
        device_map = infer_auto_device_map(base_model, max_memory={0: "60GiB", "cpu": "96GiB"})
        base_model = load_checkpoint_and_dispatch(
            base_model, 
            LLAMA_PATH, 
            device_map=device_map
        )
    elif "alpaca-7b-zs" in model:
        args["MAX_LEN"]=2048
        tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
        base_model = LlamaForCausalLM.from_pretrained(
            "chavinlo/alpaca-native",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map='auto',
        )
    elif "vicuna-13b-zs" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-13b")
        base_model = AutoModelForCausalLM.from_pretrained(
            "eachadea/vicuna-13b",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map='auto',
        )
    elif "gpt4-x-alpaca-zs" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("chavinlo/gpt4-x-alpaca")
        base_model = AutoModelForCausalLM.from_pretrained("chavinlo/gpt4-x-alpaca", device_map="auto", load_in_8bit=True)
    elif "topp-zs" in model:
        args["MAX_LEN"]=512
        tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
        base_model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    elif "flan-t5-xxl-zs" in model:
        args["MAX_LEN"]=512
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    elif "flan-ul2-zs" in model:
        args["MAX_LEN"]=512
        base_model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto")                                                                 
        tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    elif "galpaca-30b-zs" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("GeorgiaTechResearchInstitute/galpaca-30b", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained("GeorgiaTechResearchInstitute/galpaca-30b")
    elif "opt-iml-max-30b-zs" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", use_fast=False, padding_side='left')
        base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-max-30b", device_map="auto", torch_dtype=torch.float16)
    elif model == "doduo":
        print("Loading Doduo model")
        args_doduo = argparse.Namespace
        #args.model = "wikitable" (121 classes?)
        args_doduo.model = "viznet" #78 classes (Sherlock labels)
        base_model = Doduo(args_doduo)
        tokenizer = template = pt = MAX_LEN = params = None
    else:
        print("Sorry, I don't recognize model name {}. Please try again.".format(model))
    if any(["flan-t5-xxl-zs" in model, "topp-zs" in model, "flan-ul2-zs" in model, \
            "-chorus" in model, "-korini" in model, "-noisy" in model, \
            "-short" in model, "-inverted" in model]):
        template = """{instruction}"""
    else:
        template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction: 
        {instruction}

        Answer:"""
    pt = PromptTemplate(template=template, input_variables=["instruction"])
    if "llama" in model:
        params = {
            'max_new_tokens': 6,
            'do_sample': True,
            'temperature': 0.2,
            'top_p': 0.8,
            'typical_p': 1,
            'repetition_penalty': 1.3,
            'encoder_repetition_penalty': 1.0,
            'top_k': 0,
            'min_length': 3,
            'no_repeat_ngram_size': 3,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': rand_seed,
        }
    else:
        params = None
    args["base_model"] = base_model
    args["tokenizer"] = tokenizer
    args["template"] = template
    args["pt"] = pt
    #Convert length from tokens to characters, leave room for model response
    args["MAX_LEN"] = args["MAX_LEN"] * EST_CHARS_PER_TOKEN - 200
    args["params"] = params
    return

def fuzzy_label_match(orig_ans, fixed_labels, session, link, prompt, lsd, model, method=["ans_contains_gt", "gt_contains_ans", "resample"], args=dict()):
    #answer is already in label set, no fuzzy match needed
    ans_n = fix_labels(orig_ans, lsd)
    res = basic_contains(ans_n, fixed_labels, method)
    if res:
        return res
    if "similarity" in method:
        ans_embedding = args["sent_model"].encode(ans_n)
        lbl_embeddings = args["sent_model"].encode(fixed_labels)
        sims = {lbl : util.pytorch_cos_sim(ans_embedding, le) for lbl, le in zip(fixed_labels, lbl_embeddings)}
        return max(sims, key=sims.get)
    if "resample" in method:
        #fuzzy label matching strategy
        for k in range(2,6):
            if "gpt" in model:
                ans_n = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0 + k/10,
                ).choices[0]['message']['content'].lower()
            elif model in ["llama-zs", "opt-iml-30b-zs", "ArcheType-llama", "ArcheType-llama-oc"]:
                set_pipeline(k=k, args=args)
                ans_n = args['llm_chain'].run(prompt)
            elif model in ["topp-zs", "flan-t5-xxl-zs", "flan-ul2-zs"]:
                ans_n = get_topp_resp(prompt, k, args)
            else:
                top_p = args['params']['top_p']
                temp = args['params']['temperature']
                ans_n = call_llama_model(session, link, prompt, lsd, {'no_repeat_ngram_size' : 1, 'top_p' : top_p - (0.1 * k), 'temperature' : 0.9})
                args['params']['top_p'] = top_p
                args['params']['temperature'] = temp
            res = basic_contains(ans_n, fixed_labels, method)
            if res:
                return res
    return 'text'

def get_sherlock_resp(df, gt_df, prompt_dict, model, label_indices, base_prompt, lsd, args):
  isd4 = "d4" in lsd['name']
#   if "sherlock" in model:
#     model = sherlock_model
#     data_m = pd.Series(df[label_indices].astype(str).T.values.tolist())
#     extract_features(
#         "../temporary.csv",
#         data_m
#     )
#     feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)
#     predicted_labels = model.predict(feature_vectors, "sherlock")
#     iter_len = len(data_m)
  if "doduo" in model:
    data_m = df[label_indices]
    annot_m = args["base_model"].annotate_columns(data_m)
    predicted_labels = annot_m.coltypes
    iter_len = len(predicted_labels)
  predicted_labels_dict = {i : sherlock_to_cta.get(predicted_labels[i], [predicted_labels[i]]) for i in range(iter_len)}
  
  for idx, label_idx in zip(range(iter_len), label_indices):
    prompt = base_prompt + "_" + str(label_idx)
    if isd4:
        ans = predicted_labels[0]
        label = [s.lower() for s in lsd['d4_map'][gt_df]]
    else:
        gt_row = gt_df[gt_df['column_index'] == label_idx]
        if len(gt_row) != 1:
          continue
        label = fix_labels(gt_row['label'].item(), lsd)
        ans = [fix_labels(item, lsd) for item in predicted_labels_dict[idx]]
    if isd4:
        res = ans in label
    else:
        assert isinstance(ans, list), "ans should be a list"
        res = label in ans
    ans_dict = {"response" : ans, "context" : None, "ground_truth" : label, "correct" : res, "orig_model_label" : predicted_labels[idx]}
    prompt_dict[prompt] = ans_dict
  return prompt