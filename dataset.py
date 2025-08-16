import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GestureDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row['video_id'])
        label = int(row['label_id'])

        frames_path = os.path.join(self.root_dir, video_id)
        frame_files = sorted(os.listdir(frames_path))[:self.max_frames]

        images = []
        for f in frame_files:
            img_path = os.path.join(frames_path, f)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Выбираем только центральный кадр
        return images[len(images)//2], label
