#! /bin/bash
sudo -s
# export CURRENT_USER=$(whoami)
git clone https://github.com/penfever/archetype/
#export SOTAB_PATH="/home/$CURRENT_USER"
cd archetype
export SOTAB_PATH=$(pwd)
# export PATH=$PATH:/usr/lib
python3 -m pip install -r requirements.txt
curl -O https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Test.zip
unzip -q CTA_Test.zip
mkdir results
sed -i "s|ARCHETYPE_PATH = \"/home/bf996/archetype\"|ARCHETYPE_PATH = \"$(pwd)\"|" src/const.py
export SAVE_PATH="results/flan-t5-xxl-zs-choruslike.json"
# export PYTHONPATH="/usr/local/lib/python3.9/"
# export SOTAB_PATH="."
python src/run.py --model_name="flan-t5-xxl-zs-chorusprompt" --save_path="$SAVE_PATH" --input_files="$SOTAB_PATH/Test" --input_labels="$SOTAB_PATH/CTA_test_gt.csv" --label_set="SOTAB-27" --method similarity simple_random_sampling --results --response && gsutil cp "$SAVE_PATH" "gs://tabular-icl/archetype/$SAVE_PATH"
ZONE=$(gcloud compute instances list --filter=$(hostname) --format 'csv[no-heading](zone)')
sudo gcloud compute instances delete $(hostname) --quiet --zone="$ZONE"