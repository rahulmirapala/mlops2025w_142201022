import wandb
import pandas as pd
import re
from datasets import load_dataset
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter

ABSTAIN = -1
ORGANIZATION = 0
MISC = 1

@labeling_function()
def lf_year_misc(x):
    for token in x.tokens:
        if re.match(r"^(19|20)\d{2}$", token):
            return MISC
    return ABSTAIN

@labeling_function()
def lf_org_suffixes(x):
    suffixes = {"Inc.", "Corp.", "Ltd."}
    if any(token in suffixes for token in x.tokens):
        return ORGANIZATION
    return ABSTAIN

run = wandb.init(project="Q3-weak-supervision-ner", job_type="majority_voter_eval")
dataset = load_dataset(path="conll2003", split="train")
df = dataset.to_pandas()
tag_names = dataset.features['ner_tags'].feature.names

def create_unified_ground_truth(row):
    has_org = False
    has_misc = False
    for tag_id in row.ner_tags:
        tag_name = tag_names[tag_id]
        if "ORG" in tag_name:
            has_org = True
        elif "MISC" in tag_name:
            has_misc = True
    
    if has_org:
        return ORGANIZATION
    elif has_misc:
        return MISC
    else:
        return ABSTAIN

df['y_true'] = df.apply(create_unified_ground_truth, axis=1)

lfs = [lf_year_misc, lf_org_suffixes]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df)

majority_model = MajorityLabelVoter()

has_at_least_one_label = (L_train != ABSTAIN).any(axis=1)
df_labeled = df[has_at_least_one_label]
L_train_labeled = L_train[has_at_least_one_label]

majority_voter_score = majority_model.score(
    L=L_train_labeled, Y=df_labeled.y_true, tie_break_policy="random"
)
print(majority_voter_score)
accuracy = majority_voter_score["accuracy"]
print(f"\nMajority Label Voter Accuracy (on labeled data): {accuracy:.4f}")

wandb.summary["majority_voter_accuracy"] = accuracy
wandb.log({"majority_voter_accuracy": accuracy})

run.finish()