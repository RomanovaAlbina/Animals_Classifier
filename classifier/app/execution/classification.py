import torch
from torch import nn
from torchvision import models, transforms


class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.network = models.resnet50(pretrained=False)
        for param in self.network.parameters():
            param.requires_grad = False
        f = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True), nn.Linear(f, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, xb):
        x = self.network(xb)
        x = self.sigmoid(x)
        return x


class AnimalsClassifier:
    _IMGTOTENSOR = transforms.ToTensor()
    _WEIGHTS_PATH = "weights/model.pth"
    _DEVICE = "cpu"
    _CLASSIFIER_THRESHOLD = 0.7

    def __init__(self):
        self.network = ClassifierModel()
        self.network.load_state_dict(torch.load(self._WEIGHTS_PATH, map_location=torch.device(self._DEVICE)))
        # self.network.to(self._DEVICE)
        self.network.eval()

    def classify_image(self, img: bytes) -> bool:
        input_tensor = self._IMGTOTENSOR(img).unsqueeze(0)
        with torch.no_grad():
            outputs = self.network.forward(input_tensor.to(self._DEVICE))
        threshold = torch.sigmoid(outputs)
        return bool(threshold.cpu().numpy() >= self._CLASSIFIER_THRESHOLD)
