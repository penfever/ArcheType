from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from sentence_transformers import SentenceTransformer
import torch
import openai
import requests
from retry import retry
import random
import torch
import numpy as np
import re

import os
import argparse
import gc
import time

try:
    from .const import *
    from .data import *
    from .match import *
except ImportError:
    from const import *
    from data import *
    from match import *

def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    import torch
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)

def seed_all(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    elif any(["speechless-llama2" in model, "llama-zs" in model, "opt-iml-max-30b-zs" in model, "ArcheType-llama" in model, "ArcheType-llama-oc" in model]):
        end_of_sentence = prompt[-15:]
        try:
            orig_ans = args["llm_chain"].run(prompt)
        except:
            set_pipeline(k=1, args=args)
            orig_ans = args["llm_chain"].run(prompt)
        if "speechless-llama2" in model:
            orig_ans = orig_ans.split(end_of_sentence)[-1]
    elif "internlm" in model:
        orig_ans = get_internlm_resp(prompt, 1, args)
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
    # print("starting generate")
    inputs = args["tokenizer"].encode(prompt, return_tensors="pt").cuda()
    # inputs = inputs[:,:args["MAX_LEN"]-100]
    outputs = args["base_model"].generate(inputs, 
                                  max_length=args["MAX_LEN"],
                                  temperature=0.1*k,
                                  top_p=0.90-(0.1 * k),
                                  do_sample=True,
                                  repetition_penalty=1.3
                                  )
    orig_ans = args["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    # print("finished generate")
    return orig_ans

def get_internlm_resp(prompt, k, args):
    end_of_sentence = prompt[-15:]
    inputs = args["tokenizer"].encode(prompt, return_tensors="pt").cuda()
    #inputs = inputs[:,:args["MAX_LEN"]-100]
    outputs = args["base_model"].generate(inputs, 
                                  max_length=args["MAX_LEN"],
                                  temperature=0.1*k,
                                  top_p=0.90-(0.1 * k),
                                  do_sample=True,
                                  repetition_penalty=1.3
                                  )

    orig_ans = args["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    split_sent = orig_ans.split(end_of_sentence)
    orig_ans = split_sent[-1]
    return orig_ans

@retry(Exception, tries=3, delay=3)
def get_model_resp(lsd: dict, context : list, ground_truth : str, prompt_dict : dict, link : str, response = True, session=None, cbc=None, model="llama", limited_context=None, method = ["ans_contains_gt", "gt_contains_ans", "resample"], args = dict()):
  ground_truth = fix_labels(ground_truth, lsd)
  all_labels = set([fix_labels(s, lsd) for s in lsd['label_set']])
  isd4 = "d4" in lsd['name']
  ispubchem = "pubchem" in lsd['name']
  if isd4:
    target_labels = set(lsd['label_set'])
    drop_labels = set(['school-dbn', 'school-number', 'permit-types', 'us-state', 'school-grades', 'other-states', 'plate-type', 'borough'])
    target_labels = target_labels - drop_labels
    fixed_labels = sorted(list(set([fix_labels(s, lsd) for s in target_labels])))
  elif ispubchem:
    target_labels = set(lsd['label_set'])
    drop_labels = set(['Concept Broader Term', 'Journal ISSN', 'InChI (International Chemical Identifier)', "Book ISBN", 'MD5 Hash'])
    target_labels = target_labels - drop_labels
    fixed_labels = sorted(list(set([fix_labels(s, lsd) for s in target_labels])))
  elif "hierarchical" in method and not isd4:
    dtype = get_base_dtype(limited_context)
    fixed_labels = sotab_top_hier[dtype]
  elif (lsd['name'] in ['context_labels', 'context_labels_trim', 'context_labels_small']) and \
    "gpt" not in model:
      if len(limited_context) > 1 and all([re.sub('[\W_]+', '', s).isdigit() for s in limited_context]):
        if args.get("numeric_labels", -1) == -1:
            args['numeric_labels'] = sorted(list(set([fix_labels(s, lsd) for s in numeric_labels])), key=len, reverse=True)
        fixed_labels = args['numeric_labels']
      else:
        if args.get("non_numeric_labels", -1) == -1:
            num_labs = set([fix_labels(s, lsd) for s in always_numeric_labels])
            all_labels = set([fix_labels(s, lsd) for s in lsd['label_set']])
            args['non_numeric_labels'] = sorted(list(all_labels.difference(num_labs)))
        fixed_labels = args['non_numeric_labels']
  else:
    target_labels = set(lsd['label_set'])
    lsd['label_set'] = lsd['label_set'] + all_labels
    fixed_labels = sorted(list(set([fix_labels(s, lsd) for s in target_labels])), key=len, reverse=True)

  context_labels = ", ".join(fixed_labels)

  if "check_labels" in method:
    assert ground_truth in fixed_labels, f"Ground truth {ground_truth} not in label set {fixed_labels}"
  if any(["speechless-llama2" in model, "llama-zs" in model, "opt-iml-30b-zs" in model, "ArcheType-llama" in model, "ArcheType-llama-oc" in model]):
    set_pipeline(k=1, args=args)
  prompt = prompt_context_insert(context_labels, context, args["MAX_LEN"], model, args)
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
    if args['rules']:
        orig_ans = apply_basic_rules(limited_context, None, lsd)
    else:
        orig_ans = None
    if orig_ans is None:
        #model query
        orig_ans = query_correct_model(model, prompt, context_labels, context, session, link, lsd, args)
        
    #special cases
    if args['rules']:
        # ---------- d4 ------
        if orig_ans == 'abbreviation of agency' and any(len(s) > 15 for s in limited_context):
            ans_n = "nyc agency name"
        # ---------- pubchem ------
        elif orig_ans == 'patent title' and any(len(s) > 1000 for s in limited_context):
            ans_n = "abstract for patent"
    # ------------------------ 2step ------------------------
    if orig_ans == 'article' and "2step" in lsd['name']:
        context_labels = '2step'
        prompt = prompt_context_insert(context_labels, context, args["MAX_LEN"], model, args)
        orig_ans = query_correct_model(model, prompt, context_labels, context, session, link, lsd, args)
        orig_ans = 'article from ' + orig_ans
        ans_n = orig_ans.lower()
    # ------------------------ 2step ------------------------
    #hierarchical matching logic
    else: 
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
    if orig_ans.lower() not in all_labels:
        # print(f"Original answer was {orig_ans.lower()}")
        # print(f"{len(all_labels)} labels are {all_labels}")
        # print("Fuzzy matching")
        ans_n = fuzzy_label_match(orig_ans, fixed_labels, session, link, prompt, lsd, model, method=method, args=args).lower()
    else:
        ans_n = orig_ans.lower()
  if "skip-eval" in method:
    ans_n = None
  res = (ans_n == ground_truth)
  ans_dict = {"response" : ans_n, "context" : context, "ground_truth" : ground_truth, "correct" : res, "original_model_answer" : orig_ans}
  prompt_dict[prompt] = ans_dict
  return prompt

def get_sent_model(args):
    args["sent_model"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return

def set_pipeline(k=1, args=None):
    if getattr(args['tokenizer'], 'pad_token_id', None) is None:
        pad_token_id = args['tokenizer'].eos_token_id
    else:
        pad_token_id = args['tokenizer'].pad_token_id
    if args.get("params", -1) == -1 or args["params"] is None:
        args["params"] = dict()
        args['params']['max_new_tokens'] = args['params'].get('max_new_tokens', 128)
        args['params']['do_sample'] = True
        args['params']['typical_p'] = 1
        args['params']['repetition_penalty'] = 1.3
        args['params']['encoder_repetition_penalty'] = 1.0
        args['params']['top_k'] = 0
        args['params']['min_length'] = 3
        args['params']['no_repeat_ngram_size'] = 3
        args['params']['num_beams'] = 1
        args['params']['penalty_alpha'] = 0
        args['params']['length_penalty'] = 1
        args['params']['early_stopping'] = False
        args['params']['seed'] = args["rand_seed"]

    args['params']['temperature'] = 0.5*k
    args['params']['top_p'] = 0.8 - (0.1 * k)
    args["pipe"] = pipeline(
        "text-generation",
        model=args["base_model"], 
        tokenizer=args["tokenizer"], 
        max_length=args["MAX_LEN"],
        temperature=args['params']['temperature'],
        top_p=args['params']['top_p'],
        do_sample=args['params']['do_sample'],
        repetition_penalty=args['params']['repetition_penalty'],
        pad_token_id=pad_token_id,
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
    if "speechless-llama2" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("uukuguy/speechless-llama2-13b")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model = AutoModelForCausalLM.from_pretrained("uukuguy/speechless-llama2-13b", torch_dtype=torch.float16, load_in_8bit=True, device_map="auto")        
    elif "llama" in model: 
        LLAMA_PATH = args["model_path"]
        args["MAX_LEN"]=2048
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)
        if "ArcheType-llama" in model:
            base_model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH,
                                                torch_dtype=torch.float16,
                                                load_in_8bit=True,
                                                device_map='auto')
        else:
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
    elif "internlm-20b" in model:
        args["MAX_LEN"]=2048
        tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-20b", trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained("internlm/internlm-20b", trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
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
        tokenizer = AutoTokenizer.from_pretrained("GeorgiaTechResearchInstitute/galpaca-30b")
        base_model = AutoModelForCausalLM.from_pretrained("GeorgiaTechResearchInstitute/galpaca-30b", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    elif "opt-iml-max-30b-zs" in model:
        args["MAX_LEN"] = 2048
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", use_fast=False, padding_side='left')
        base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-max-30b", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
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
            "internlm" in model, \
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
            'max_new_tokens': 128,
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
            'seed': args["rand_seed"],
        }
    else:
        params = None
    args["base_model"] = base_model
    args["tokenizer"] = tokenizer
    args["template"] = template
    args["pt"] = pt
    #Convert length from tokens to characters, leave room for model response
    # args["MAX_LEN"] = args["MAX_LEN"] * EST_CHARS_PER_TOKEN - 300
    args["params"] = params
    return

# def cutoff_prompt_length(prompt, max_len):
#     if len(prompt) > 20000:
#         prompt = prompt[:20000] + "Response: "
#     approx_cutoff = max_len-1 // 3
#     word_split = prompt.split(" ")
#     if len(word_split) > approx_cutoff:
#         word_split = word_split[:approx_cutoff] + ["Response: "]
#         return " ".join(s for s in word_split if len(s) < 30)
#     else:
#         return prompt

def fuzzy_label_match(orig_ans, fixed_labels, session, link, prompt, lsd, model, method=["ans_contains_gt", "gt_contains_ans", "resample"], args=dict()):
    #basic_contains checks whether is already in label set; depending on the options passed in, it may also check for contains relationship
    ans_n = fix_labels(orig_ans, lsd)
    res = basic_contains(ans_n, fixed_labels, method)
    if res:
        return res
    #if not found, try similarity matching (if in method)
    if "similarity" in method:
        ans_embedding = args["sent_model"].encode(ans_n)
        args["lbl_embeddings"] = args["sent_model"].encode(fixed_labels)
        lbl_embeddings = args["lbl_embeddings"]
        sims = {lbl : util.pytorch_cos_sim(ans_embedding, le) for lbl, le in zip(fixed_labels, lbl_embeddings)}
        return max(sims, key=sims.get)
    #if not found, try resampling (if in method); these are usually mutually exclusive
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
            elif "internlm" in model:
                ans_n = get_internlm_resp(prompt, k, args)
            elif any(["speechless-llama2" in model, "llama-zs" in model, "opt-iml" in model, "ArcheType-llama" in model, "ArcheType-llama-oc" in model]):
                # prompt = cutoff_prompt_length(prompt, args["MAX_LEN"])
                end_of_sentence = prompt[-15:]
                set_pipeline(k=k, args=args)
                ans_n = args['llm_chain'].run(prompt)
                if "speechless-llama2" in model:
                    ans_n = ans_n.split(end_of_sentence)[-1]
            elif any(["topp-zs" in model, "flan-t5-xxl-zs" in model, "flan-ul2-zs" in model]):
                ans_n = get_topp_resp(prompt, k, args)
            else:
                print("Running default (local saved checkpoint) llama resampling -- THIS SHOULD NOT HAPPEN if you are running zero-shot models, please check model name")
                top_p = args['params']['top_p']
                temp = args['params']['temperature']
                ans_n = call_llama_model(session, link, prompt, lsd, {'no_repeat_ngram_size' : 1, 'top_p' : top_p - (0.1 * k), 'temperature' : 0.9}, args)
                args['params']['top_p'] = top_p
                args['params']['temperature'] = temp
            res = basic_contains(ans_n, fixed_labels, method)
            if res:
                return res
    # Finally, return default answer
    default_ans = fix_labels(lsd['label_set'][-1], lsd)
    return default_ans

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
