import torch
import torch.nn.functional as F
from torch import nn


# Define a highly advanced and skilled loss function for deepfake generating
# we are aiming for the highest accuracy percentage possible, with hopes of 100%
def advanced_loss_function(real_images, fake_images):
    # The function will use a combination of MSE (Mean Squared Error) and Cross-Entropy loss functions
    # The MSE loss function will be used to calculate the difference between the real images and the fake images
    # The Cross-Entropy loss function will be used to calculate how well the fake images resemble the real images
    mse_loss = F.mse_loss(real_images, fake_images)
    cross_entropy_loss = F.cross_entropy(
        fake_images, torch.tensor([0] * fake_images.shape[0])
    )
    return mse_loss + cross_entropy_loss



def hinge_loss(real_images, fake_images):
    return F.relu(1.0 - real_images * fake_images).mean()


def triplet_loss(real_images, fake_images):
    anchor = real_images
    positive = fake_images
    negative = torch.randn_like(fake_images)
    d_pos = torch.pairwise_distance(anchor, positive)
    d_neg = torch.pairwise_distance(anchor, negative)
    return F.relu(d_pos - d_neg + 1.0).mean()


def contrastive_loss(real_images, fake_images):
    real_embeddings = F.normalize(real_images)
    fake_embeddings = F.normalize(fake_images)
    labels = torch.tensor([1] * real_images.shape[0] + [0] * fake_images.shape[0])
    similarity_matrix = torch.matmul(real_embeddings, fake_embeddings.T)
    loss_function = nn.CrossEntropyLoss()
    return loss_function(similarity_matrix, labels)


def Wasserstein_loss(real_images, fake_images):
    real_embeddings = F.normalize(real_images)
    fake_embeddings = F.normalize(fake_images)
    wasserstein_distance = torch.norm(real_embeddings - fake_embeddings)
    return wasserstein_distance


def adversarial_loss(real_images, fake_images, discriminator):
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images)
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


def GAN_loss(real_images, fake_images, discriminator):
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images)
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss
