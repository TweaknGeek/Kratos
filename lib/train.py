import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import model
import eval

def train_model(model, train_loader, validation_loader, num_epochs=10, learning_rate=0.001,
                loss_fn=nn.CrossEntropyLoss(), optimizer=optim.SGD):
    model.train()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        # evaluate the model on the validation set
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                valid_accuracy += (predicted == labels).sum().item()
        valid_loss /= len(validation_loader.dataset)
        valid_accuracy /= len(validation_loader.dataset)
        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(
            epoch+1, num_epochs, valid_loss, valid_accuracy))
    return model

def adjust_learning_rate(optimizer, epoch, initial_lr=0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = initial_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_data(batch_size=16, num_workers=4):
    # Data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(root='./data/{}'.format(x), transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    n_classes = len(class_names)
    return dataloaders, n_classes
