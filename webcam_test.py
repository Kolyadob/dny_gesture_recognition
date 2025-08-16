# import cv2
# import torch
# from model import SimpleCNN
# from torchvision import transforms
# from PIL import Image

# LABELS = [
#     'Doing other things', 'Pushing Two Fingers Away', 'Drumming Fingers', 'Sliding Two Fingers Down', 'Pushing Hand Away', 'Shaking Hand', 'Pulling Two Fingers In', 'Stop Sign', 'Zooming In With Two Fingers', 'Sliding Two Fingers Up', 'Zooming Out With Two Fingers', 'Zooming In With Full Hand', 'No gesture', 'Swiping Right', 'Thumb Down', 'Rolling Hand Forward', 'Pulling Hand In', 'Zooming Out With Full Hand', 'Swiping Left', 'Rolling Hand Backward', 'Turning Hand Counterclockwise', 'Swiping Up', 'Turning Hand Clockwise', 'Sliding Two Fingers Left', 'Swiping Down', 'Thumb Up', 'Sliding Two Fingers Right'
# ]

# model = SimpleCNN(num_classes=27)
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# device = torch.device("cpu")
# model.to(device)

# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.ToTensor()
# ])

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(img_tensor)
#         _, pred = torch.max(output, 1)

#     label = LABELS[pred.item()]
#     cv2.putText(frame, f'Gesture: {label}', (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.imshow('Gesture Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
from quick_model import SimpleCNN
from torchvision import transforms
from PIL import Image

LABELS = [
    'Doing other things', 'Pushing Two Fingers Away', 'Drumming Fingers',
    'Sliding Two Fingers Down', 'Pushing Hand Away', 'Shaking Hand',
    'Pulling Two Fingers In', 'Stop Sign', 'Zooming In With Two Fingers',
    'Sliding Two Fingers Up', 'Zooming Out With Two Fingers',
    'Zooming In With Full Hand', 'No gesture', 'Swiping Right', 'Thumb Down',
    'Rolling Hand Forward', 'Pulling Hand In', 'Zooming Out With Full Hand',
    'Swiping Left', 'Rolling Hand Backward', 'Turning Hand Counterclockwise',
    'Swiping Up', 'Turning Hand Clockwise', 'Sliding Two Fingers Left',
    'Swiping Down', 'Thumb Up', 'Sliding Two Fingers Right'
]

device = torch.device('cpu')
model = SimpleCNN(num_classes=27)
model.load_state_dict(torch.load('model_quick.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((50, 88)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    label = LABELS[pred.item()]
    cv2.putText(frame, f'Gesture: {label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
