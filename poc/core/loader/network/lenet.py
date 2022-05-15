import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes),
            #act(),
            #nn.Linear(256, 100)
        )
        self.feature = None
        
    def forward(self, x):
        out = self.body(x)
        self.feature = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(self.feature)
        return out

    def extract_feature(self):
        return self.feature