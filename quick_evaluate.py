from dataset import GestureDataset
from quick_model import SimpleCNN
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from sklearn.metrics import classification_report
from tqdm import tqdm  # <-- импортируем tqdm

transform = transforms.Compose([
    transforms.Resize((50, 88)),
    transforms.ToTensor()
])

val_dataset = GestureDataset('/mnt/d/diplom_vkr/validation.csv',
                             '/mnt/d/diplom_vkr/20bn-jester-v1/', transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cpu')
model = SimpleCNN(num_classes=27)
state_dict = torch.load('model_quick.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))
