from dataset import GestureDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from sklearn.metrics import classification_report

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

val_dataset = GestureDataset('/mnt/d/diplom_vkr/validation.csv', '/mnt/d/diplom_vkr/20bn-jester-v1/', transform)
val_loader = DataLoader(val_dataset, batch_size=32)

model = SimpleCNN(num_classes=27)
model.load_state_dict(torch.load('model_quick.pth'))
model.eval()

device = torch.device("cpu")
model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))
