import json
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import random
from itertools import product

torch.manual_seed(42)
random.seed(42)

def load_config(filepath):
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.toml'):
        with open(filepath, 'r') as f:
            return toml.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .toml")

def get_model(model_name, pretrained=True, num_classes=10):
    weights = None
    if pretrained:
        weights_map = {
            "resnet34": getattr(models, "ResNet34_Weights", None),
            "resnet50": getattr(models, "ResNet50_Weights", None),
            "resnet101": getattr(models, "ResNet101_Weights", None),
            "resnet152": getattr(models, "ResNet152_Weights", None),
        }
        if model_name in weights_map and weights_map[model_name] is not None:
            weights = getattr(weights_map[model_name], "DEFAULT", None)
        else:
            weights = 'IMAGENET1K_V1'

    if model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    elif model_name == "resnet101":
        model = models.resnet101(weights=weights)
    elif model_name == "resnet152":
        model = models.resnet152(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, dataloaders, params):
    """Trains and evaluates the model with a given set of parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params.get('momentum', 0.9))
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    print(f"Training with LR: {params['learning_rate']}, Optimizer: {params['optimizer']}, Batch Size: {params['batch_size']}")

    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, model

def main():
    data_model_config = load_config('data_model.json')
    model_params_config = load_config('config.toml')
    hyperparams_config = load_config('hyperparameters.json')

    datasource_info = data_model_config['datasource']
    num_samples = datasource_info['num_samples']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading CIFAR-10 dataset (will download if not present)...")
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("Dataset loaded.")

    train_indices = random.sample(range(len(full_trainset)), num_samples)
    train_subset = Subset(full_trainset, train_indices)
    
    num_test_samples = int(num_samples * (1 - datasource_info.get('train_split', 0.8)))
    test_indices = random.sample(range(len(full_testset)), num_test_samples)
    test_subset = Subset(full_testset, test_indices)
    
    print(f"\nUsing a subset of {len(train_subset)} training samples and {len(test_subset)} testing samples.")


    # based on whether hyperparameters.json contains grid search parameters.
    
    # Check if we should do hyperparameter tuning (if hyperparameters.json has multiple values)
    should_tune = any(isinstance(v, list) and len(v) > 1 for v in hyperparams_config.values())
    
    if should_tune:
        print("\n--- Starting Hyperparameter Tuning ---")
        # Normalize key names: momentums -> momentum for consistency
        grid_params = hyperparams_config.copy()
        if 'momentums' in grid_params:
            grid_params['momentum'] = grid_params.pop('momentums')
        
        keys = list(grid_params.keys())
        values_product = list(product(*[grid_params[k] for k in keys]))

        best_accuracy = -1
        best_params = None
        best_model_name = None

        for model_config in data_model_config['models_to_train']:
            model_name = model_config['name']
            print(f"\n--- Tuning for {model_name} ---")
            
            for values in values_product:
                current_params = model_params_config.get('model_params', {}).get(model_name, {}).copy()
                combo = dict(zip(keys, values))
                if 'learning_rates' in combo:
                    combo['learning_rate'] = combo.pop('learning_rates')
                if 'optimizers' in combo:
                    combo['optimizer'] = combo.pop('optimizers')
                current_params.update(combo)
                current_params.setdefault('epochs', 2)
                current_params.setdefault('batch_size', 32)
                
                print(f"\nTrying params: {current_params}")
                
                dataloaders = {
                    'train': DataLoader(train_subset, batch_size=current_params['batch_size'], shuffle=True),
                    'test': DataLoader(test_subset, batch_size=current_params['batch_size'], shuffle=False)
                }

                model = get_model(model_name, pretrained=model_config['pretrained'], num_classes=model_config['num_classes'])
                accuracy, _ = train_model(model, dataloaders, current_params)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = current_params
                    best_model_name = model_name
        
        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Best Model Found: {best_model_name}")
        print(f"Best Test Accuracy: {best_accuracy:.2f}%")
        print(f"Best Parameters: {best_params}")
    
    else:
        print("\n--- Starting Standard Model Training ---")
        for model_config in data_model_config['models_to_train']:
            model_name = model_config['name']
            print(f"\n--- Processing model: {model_name} ---")

            params = model_params_config['model_params'][model_name]
            
            dataloaders = {
                'train': DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True),
                'test': DataLoader(test_subset, batch_size=params['batch_size'], shuffle=False)
            }

            model = get_model(model_name, pretrained=model_config['pretrained'], num_classes=model_config['num_classes'])
            
            print(f"Training {model_name} with parameters from config.toml...")
            accuracy, _ = train_model(model, dataloaders, params)
            print(f"Final Test Accuracy for {model_name}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()