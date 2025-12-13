import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import cv2

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


def compress_frame(img, quality=70):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)

    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    return decoded_img


def obtain_frames(video_p, output_p, frames_num) -> None:
    current_frame = 0
    video_path = video_p
    output_path = output_p

    cap = cv2.VideoCapture(video_path)

    while(True):
        success, frame = cap.read()
        current_frame += 1
        if not success or current_frame == frames_num + 1:
            break

        cv2.imshow('image', frame)

        compressed_img = compress_frame(frame)
        cv2.imwrite(f"{output_path}/frame{current_frame}.jpg", compressed_img)

        print(f"Saved frame #{current_frame}")
        cv2.waitKey(1)

    cap.release()

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class StudyHabitClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = x.flatten(1)

        output = self.classifier(x)
        return output


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_folder = "../data"
full_dataset = VideoDataset(data_folder, transform)
dataset_size = len(full_dataset)
lengths = [int(0.8 * dataset_size), int(0.2 * dataset_size)]

train_dataset, valid_dataset = random_split(full_dataset, lengths)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

num_epoch = 5
train_losses, val_losses = [], []

device = torch.device("mps")
print(device)

model = StudyHabitClassifier(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in tqdm(range(num_epoch), desc="Overall Training Loop"):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training Loop"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation Loop"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(valid_dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epoch} - Train Loss: {train_loss} - Validation Loss: {val_loss}")


def visualize_predictions(model, loader, num_images=12):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(12,8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1

                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.axis('off')

                predicted_class = full_dataset.classes[preds[j]]
                actual_class = full_dataset.classes[labels[j]]

                color = 'green' if predicted_class == actual_class else 'red'
                ax.set_title(f'Predicted: {predicted_class}\nActual: {actual_class}', color=color)

                ax.imshow(inputs.cpu().data[j].permute(1, 2, 0))

                if images_so_far == num_images:
                    plt.show()
                    return

visualize_predictions(model, train_loader)
