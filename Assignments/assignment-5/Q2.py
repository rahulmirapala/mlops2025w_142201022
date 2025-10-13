import wandb
import pandas as pd
import re
from datasets import load_dataset
from snorkel.labeling import labeling_function, PandasLFApplier
from sklearn.metrics import accuracy_score

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
    suffixes = ["Inc.", "Corp.", "Ltd."]
    if any(token in suffixes for token in x.tokens):
        return ORGANIZATION
    return ABSTAIN

run = wandb.init(project="Q2-weak-supervision-ner", job_type="lf_analysis")
# Load data
dataset = load_dataset(path = "conll2003", split="train")
df = dataset.to_pandas()
tag_names = dataset.features['ner_tags'].feature.names
# print(tag_names)

def has_entity(row, entity_short_name):
    entity_tags = {f"B-{entity_short_name}", f"I-{entity_short_name}"}
    for tag_id in row.ner_tags:
        if tag_names[tag_id] in entity_tags:
            return True
    return False

df['y_org'] = df.apply(lambda row: ORGANIZATION if has_entity(row, "ORG") else ABSTAIN, axis=1)

df['y_misc'] = df.apply(lambda row: MISC if has_entity(row, "MISC") else ABSTAIN, axis=1)

# Apply the LFs to the dataframe
lfs = [lf_year_misc, lf_org_suffixes]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df)

lf_eval_map = {
    "lf_year_misc":    {"labels": L_train[:, 0], "ground_truth": df.y_misc},
    "lf_org_suffixes": {"labels": L_train[:, 1], "ground_truth": df.y_org},
}

for lf_name, eval_data in lf_eval_map.items():
    lf_labels = eval_data["labels"]
    ground_truth = eval_data["ground_truth"]
    
    # Calculate Coverage
    coverage = (lf_labels != ABSTAIN).sum() / len(lf_labels)
    # Calculate Accuracy on the covered subset
    active_indices = lf_labels != ABSTAIN
    if active_indices.sum() > 0:
        accuracy = accuracy_score(ground_truth[active_indices], lf_labels[active_indices])
    else:
        accuracy = 0.0

    print(f"\n--- {lf_name} ---")
    print(f"Coverage: {coverage:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    wandb.summary[f"{lf_name}_coverage"] = coverage
    wandb.summary[f"{lf_name}_accuracy"] = accuracy

    wandb.log({
        f"{lf_name}_coverage": coverage,
        f"{lf_name}_accuracy": accuracy
    })

run.finish()