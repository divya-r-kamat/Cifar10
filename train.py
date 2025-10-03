from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import importlib.util
import sys
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

horizontalflip_prob = 0.5
shiftscalerotate_prob = 0.5
rotate_limit = 15
cutout_prob = 0.5
num_holes = 1
mean = (0.4919, 0.4829, 0.4467)
std  = (0.2444, 0.2408, 0.2582)

class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        img = np.array(img)  # PIL -> numpy
        return self.aug(image=img)["image"]


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss = 0
    correct = 0
    processed = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Train loss={loss.item():0.4f} batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    train_losses.append(loss/len(train_loader))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Train MNIST model from any model file')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (e.g., model1.py)')
    



def load_model_from_file(model_file):
    """
    Dynamically load a model from a Python file
    
    Args:
        model_file: Path to the Python file containing the model
        
    Returns:
        Model class from the file
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found!")
    
    # Get the module name from file path
    module_name = Path(model_file).stem
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Find the model class (usually named 'Net' or 'Model')
    model_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, torch.nn.Module) and 
            obj != torch.nn.Module):
            model_class = obj
            break
    
    if model_class is None:
        raise ValueError(f"No PyTorch model class found in {model_file}")
    
    print(f"✅ Loaded model class '{model_class.__name__}' from {model_file}")
    return model_class


def main():
    parser = argparse.ArgumentParser(description='Train MNIST model from any model file')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (e.g., model1.py)')

    parser.add_argument('--optimizer', type=str, required=True, help="Name of optimizer variable")
    parser.add_argument('--scheduler', type=str, required=True, help="Name of scheduler variable")
    parser.add_argument('--train_transforms', type=str, required=True, help="Train transforms variable name")
    parser.add_argument('--test_transforms', type=str, required=True, help="Test transforms variable name")
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to train (default: 15)')


    args = parser.parse_args()
    model_class = load_model_from_file(args.model)
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda" if cuda else "cpu")
    model = model_class().to(device)

    # --- fetch optimizer/scheduler objects from globals ---

    optimizer = eval(args.optimizer, {"optim": optim, "model": model})
    scheduler = eval(args.scheduler, {"optim": optim, "optimizer": optimizer})
    train_transforms = eval(args.train_transforms)
    test_transforms = eval(args.test_transforms)

    print("\nDownload the train and test dataset, with transforms....")
    # train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    # test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=AlbumentationsTransform(train_transforms))
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=AlbumentationsTransform(test_transforms))

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)


    print("\nModel Training....\n")

    num_epochs = 30

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

if __name__ == '__main__':
    main()    
