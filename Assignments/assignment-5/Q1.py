import wandb
import pandas as pd
from collections import Counter
from datasets import load_dataset

# 1. Initialize W&B Run
run = wandb.init(project="Q1-weak-supervision-ner", job_type="data_analysis")

dataset = load_dataset(
    path="conll2003",
    trust_remote_code=True
)

tag_names = dataset['train'].features['ner_tags'].feature.names

all_entity_counts = Counter()
total_samples = 0
split_stats = {}

for split_name in dataset.keys():
    split_data = dataset[split_name]
    df_split = pd.DataFrame(split_data)
    
    num_samples_split = len(df_split) # Number of samples
    total_samples += num_samples_split

    # Count entities for the current split.
    split_entity_counts = Counter()
    for tags in df_split['ner_tags']:
        for tag_id in tags:
            tag_name = tag_names[tag_id]
            if tag_name.startswith("B-"): # Taking the tag only starting with B to avoid duplicates
                entity_type = tag_name.split("-")[1]
                split_entity_counts[entity_type] += 1
    
    all_entity_counts.update(split_entity_counts)
    
    split_stats[split_name] = {
        'samples': num_samples_split,
        'entities': dict(split_entity_counts)
    }
    
    # Log the raw numbers for this split to the W&B summary.
    wandb.summary[f"{split_name}_samples"] = num_samples_split
    for entity, count in split_entity_counts.items():
        wandb.summary[f"{split_name}_entity_count_{entity}"] = count
    
wandb.summary["total_samples"] = total_samples
for entity, count in all_entity_counts.items():
    wandb.summary[f"total_entity_count_{entity}"] = count

# For the total dataset
overall_table = wandb.Table(
    data=[[label, val] for (label, val) in all_entity_counts.items()],
    columns=["entity", "count"]
)
wandb.log({
    "overall_entity_distribution": wandb.plot.bar(
        overall_table, "entity", "count",
        title="Overall Entity Distribution in CoNLL-2003"
    )
})

# For the test,val,train 
for split_name, stats in split_stats.items():
    split_table = wandb.Table(
        data=[[entity, count] for entity, count in stats['entities'].items()],
        columns=["entity", "count"]
    )
    
    wandb.log({
        f"{split_name}_entity_distribution": wandb.plot.bar(
            split_table, "entity", "count",
            title=f"Entity Distribution in {split_name.capitalize()} Split"
        )
    })

# 7. Finish the Run
run.finish()
