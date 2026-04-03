import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchmetrics import Precision, Recall

batch_size = 60

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
])

train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = loss_func(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

acc = Accuracy(task="multiclass", num_classes=10).to(device)
prec_metric = Precision(task="multiclass", num_classes=10, average='macro').to(device)
rec_metric = Recall(task="multiclass", num_classes=10, average='macro').to(device)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        # update metrics
        acc.update(preds, labels)
        prec_metric.update(preds, labels)
        rec_metric.update(preds, labels)

test_accuracy = acc.compute().item()
test_precision = prec_metric.compute().item()
test_recall = rec_metric.compute().item()
print(f"Test accuracy: {test_accuracy}")
print(f"Test precision (macro): {test_precision}")
print(f"Test recall (macro): {test_recall}")

torch.save(model.state_dict(), 'static/MulticlassCNN.pth')

dummy_input = torch.randn(1, 1, 28, 28).to(device)
onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)
onnx_program.save("static/MulticlassCNN.onnx")

