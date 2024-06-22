import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn
import torch.nn.functional as F

# Load the CelebA dataset
transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()])
trainset = datasets.CelebA(root='./data', download=True,
                           split='train', transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Define the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
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

    def forward(self, x):
        x = self.main(x)
        return x

# Train the generator
gen = Generator()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 100
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        input, target = data
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = gen(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Use the generator to create fake images
fake_image = gen(input)
save_image(fake_image.data, 'fake_image.png')
