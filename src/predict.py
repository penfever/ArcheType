import os
from pathlib import Path

from src.run import run
from src.model import seed_all
from src.data import get_lsd

class ArcheTypePredictor():

    def __init__(self, input_files = None, user_args = None):
        class Args:
            pass

        args = Args()

        if input_files is not None:
            self.input_files = input_files
        else:
            self.input_files = user_args.input_files

        #load default configuration settings
        args = self.get_default_config(args)

        if user_args is not None:
            for k, v in user_args.items():
                setattr(args, k, v)
        
        args = self.parse_additional_args(args, user_args)

        save_path = Path(args.save_path)
        if not os.path.exists(save_path.parent):
            os.makedirs(save_path.parent)
        
        self.args = args
        seed_all(args.rand_seed)

    def get_default_config(self, args):
        args.model_name = "flan-t5-base-zs"
        args.save_path = "./results/archetype_predict.json"
        args.method = ["ans_contains_gt", "gt_contains_ans", "resample"]
        args.results = True
        args.response = True
        args.resume = False
        args.input_files = "./table_samples/Book_5sentidoseditora.pt_September2020_CTA.json"
        args.input_labels = "skip-eval-return"
        args.label_set = "custom"
        args.custom_labels = ["text", "number", "id", "place"]
        args.stop_early = -1
        args.rand_seed = 1902582
        args.sample_size = 5
        args.link = ""
        args.summ_stats = False
        args.table_src = False
        args.other_col = False
        args.skip_short = False
        args.min_var = 0
        return args

    def parse_additional_args(self, args, user_args):
        if args.label_set == "custom":
            label_set = {"name" : "custom", "label_set" : args.custom_labels, "dict_map" : {c : c for c in args.custom_labels}, 'abbrev_map' : {c : c for c in args.custom_labels}}
        else:
            label_set = get_lsd(args.label_set)
        args.addl_args = {"MAX_LEN" : 512, 
                "model_path" : user_args.get("model_path", ""), 
                "lsd" : label_set, 
                "rules" : user_args.get("rules", True), 
                "oracle" : user_args.get("oracle", False),
                "partial_oracle" : user_args.get("partial_oracle", False),
                "input_labels" : args.input_labels,
                "return_prompt" : False,
                "k_shot" : int(user_args.get("k_shot", 0))}
        return args

    def annotate_columns(self):
        args = self.args
        df = run(
            args.model_name, 
            args.save_path,
            self.input_files,
            args.addl_args["lsd"],
            None,
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
            args.addl_args,
        )
        return df