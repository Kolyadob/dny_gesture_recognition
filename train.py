import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import GestureDataset
from model import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

train_dataset = GestureDataset('/mnt/d/diplom_vkr/train.csv', '/mnt/d/diplom_vkr/20bn-jester-v1/', transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleCNN(num_classes=27)
device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), 'model.pth')
