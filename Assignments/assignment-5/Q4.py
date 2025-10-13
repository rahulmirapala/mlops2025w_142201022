import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

config = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "epochs": 100,
    "batch_size": 64,
    "architecture": "SimpleCNN"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes):
    return SimpleCNN(num_classes).to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=config["batch_size"], shuffle=True, num_workers=2)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=config["batch_size"], shuffle=False, num_workers=2)

cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=config["batch_size"], shuffle=True, num_workers=2)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=config["batch_size"], shuffle=False, num_workers=2)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / total
    eval_acc = running_corrects / total
    return eval_loss, eval_acc

def train_model(model, train_loader, test_loader, run_name):
    run = wandb.init(
        project="Q4-weak-supervision-ner",
        config=config,
        name=run_name,
        reinit=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    model.train()

    for epoch in range(config["epochs"]):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"{run_name} | Epoch {epoch+1}/{config['epochs']}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_corrects / total

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

    run.finish()

if __name__ == "__main__":
    # Scenario A: CIFAR-100 → CIFAR-10
    model_A = build_model(num_classes=100)
    train_model(model_A, cifar100_train_loader, cifar100_test_loader, run_name="A_Train_CIFAR100")

    model_A.classifier[-1] = nn.Linear(256, 10).to(device)
    train_model(model_A, cifar10_train_loader, cifar10_test_loader, run_name="A_Finetune_on_CIFAR10")

    # Scenario B: CIFAR-10 → CIFAR-100
    model_B = build_model(num_classes=10)
    train_model(model_B, cifar10_train_loader, cifar10_test_loader, run_name="B_Train_CIFAR10")

    model_B.classifier[-1] = nn.Linear(256, 100).to(device)
    train_model(model_B, cifar100_train_loader, cifar100_test_loader, run_name="B_Finetune_on_CIFAR100")
