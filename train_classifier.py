import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, datasets
from train_autoencoder import AutoEncoder
import os
import torch.nn.functional as F

transform = transforms.Compose(
    [
        # transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

dataset = datasets.ImageFolder("train_data", transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


class CombineLayer(nn.Module):
    def __init__(self, stride):
        super(CombineLayer, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, stride=stride, padding="valid", groups=128)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=stride, padding="valid", groups=128)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=stride, padding="valid", groups=128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=stride, padding="valid", groups=128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        return x1 + x2 + x3 + x4


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.combineLayer1 = CombineLayer(1)
        self.combineLayer2 = CombineLayer(2)
        self.combineLayer3 = CombineLayer(1)
        self.combineLayer4 = CombineLayer(2)
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        x = self.combineLayer1(x)
        x = self.combineLayer2(x)
        x = self.combineLayer3(x)
        x = self.combineLayer4(x)
        x = torch.mean(x, dim=[2, 3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path + "model.pth")
        
    def load(self, path):
        self.load_state_dict(torch.load(path + "model.pth"))


def train():
    ae = AutoEncoder().cuda()
    model = Classifier().cuda()
    ae.load("model/")
    # Train Network
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 128
    for epoch in range(num_epochs):
        mean_loss = 0
        accuracy = 0
        count = 0
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = img.cuda()
            features = ae.encoder(img)
            features = features.detach()
            # ===================forward=====================
            output = model(features)
            loss = F.cross_entropy(torch.softmax(output, dim=1), label.cuda())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = mean_loss + loss.data
            accuracy = (
                accuracy + torch.sum(torch.argmax(output, dim=1) == label.cuda()).float() / 32
            )
            count = count + 1
            print(
                "\repoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}".format(
                    epoch + 1, num_epochs, mean_loss / count, accuracy / count
                ),
                end="",
            )
            if batch_idx % 100 == 0:
                path = "model_tno/model_{}_{}/".format(epoch, int(batch_idx / 100))
                os.makedirs(path)
                model.save(path)
        # ===================log========================
        print(
            "\repoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}".format(
                epoch + 1, num_epochs, mean_loss / count, accuracy / count
            ),
        )


if __name__ == "__main__":
    train()
