import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import GestureDataset
from quick_model import SimpleCNN
from tqdm import tqdm

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((50, 88)),
    transforms.ToTensor(),
])

full_train_dataset = GestureDataset('/mnt/d/diplom_vkr/train.csv',
                                    '/mnt/d/diplom_vkr/20bn-jester-v1/',
                                    transform)
subset_size = int(0.2 * len(full_train_dataset))
train_subset = Subset(full_train_dataset, list(range(subset_size)))
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)

model = SimpleCNN(num_classes=27).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

torch.save(model.state_dict(), 'model_quick.pth')
print('Модель сохранена как model_quick.pth')
