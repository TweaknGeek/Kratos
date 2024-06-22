import torch
from torchvision import datasets, transforms


# Define transformations for the CelebA dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the CelebA dataset with the defined transformations
dataset = datasets.CelebA(root='./data', download=True, split='train', transform=transform)

# Define the data loader
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Now you can easily use the trainloader to iterate over your data
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(inputs.shape)

# You can also use the transforms module to apply transformations to images
# outside of a data loader
image = datasets.CelebA.get_image_path(dataset.imgs[0])
image = datasets.CelebA.loader(image)
image = transform(image)
print(image.shape)

