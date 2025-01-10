import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch import nn, optim
from PIL import Image


# transformations (random translation, rotation, dilatation), convert to grayscale, sends to tensor, normalizes with mean=0.5 and std=0.5
transform = transforms.Compose([
    transforms.RandomAffine(degrees=(0, 20), translate=(0, 0.2), scale=(0.7, 1)), # translate, rotate, dilatate
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), # convert to tensor
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # normalize data -> easier to train
])

transformtest = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

batch_size = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load MNIST training/testing datasets and apply transformation, loaders
train_dataset = datasets.MNIST(
    root='train',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='test',
    train=False,
    transform=transformtest,
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


model = models.resnet18(num_classes=10).to(device)      # resnet18 model
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)  # adam optimizer


# train model
num_epochs = 10

for epoch in range(0, num_epochs):
  model.train()

  for i, data in enumerate(train_loader):

    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    pred = model(images)
    loss = loss_fn(pred, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 99:
      print(f"Epoch {epoch+1}, Step {i+1}, Loss = {loss.item()}")


# evaluate model
model.eval()
with torch.no_grad():

    i = 0
    total_accuracy = 0
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        test_output = model(images)
        pred_y = torch.max(test_output, 1)[1]
        total_accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
        i += 1

    accuracy = total_accuracy / i

print(f'Test Accuracy of the model on the 10000 test images: {accuracy:.3f}')
