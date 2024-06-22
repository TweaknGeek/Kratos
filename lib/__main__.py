import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from lossFunc import advanced_loss_function
from train import train_model
import model 
import eval


# Define the training configuration
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Load the CelebA dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CelebA(root='./data', download=True, split='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Define the generator
generator = nn.Sequential(
    nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
    nn.Tanh()
)

# Train the generator
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        input, target = data
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = generator(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
