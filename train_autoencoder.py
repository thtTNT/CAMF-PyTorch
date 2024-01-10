import torch
import torch.nn as nn
import torchsummary as summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision

from utils import gradient

transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder('Train_images', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),     
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding="valid"),       
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, padding="valid"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def save(self,path):
        # save encoder
        torch.save(self.encoder.state_dict(), path+'encoder.pth')
        # save decoder
        torch.save(self.decoder.state_dict(), path+'decoder.pth')
        
    def load(self,path):
        # load encoder
        self.encoder.load_state_dict(torch.load(path+'encoder.pth'))
        # load decoder
        self.decoder.load_state_dict(torch.load(path+'decoder.pth'))

def train():
    model = AutoEncoder()
    model = model.cuda()
    # Train Network
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 128
    for epoch in range(num_epochs):
        mean_loss = 0
        loss_count = 0
        for data in dataloader:
            img, _ = data
            img = img.cuda()
            # ===================forward=====================
            output = model(img)
            loss = 5 * torch.mean(torch.pow(output - img, 2)) + torch.mean(torch.pow(gradient(output) - gradient(img), 2))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss = mean_loss + loss.data
            loss_count = loss_count + 1
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, mean_loss))


def validate():
    model = AutoEncoder()
    model = model.cuda()
    model.load('model/')
    model.eval()
    for data in dataloader:
        img, _ = data
        img = img.cuda()
        # ===================forward=====================
        output = model(img)
        torchvision.utils.save_image((output + 1) / 2, 'output.png')
        torchvision.utils.save_image((img + 1) / 2, 'input.png')
        break

if __name__ == "__main__":
    model = AutoEncoder().cuda()
    summary.summary(model, (1, 256, 256))