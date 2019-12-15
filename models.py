# @Author: yusuf
# @Date:   2019-12-10T11:51:06+02:00
# @Last modified by:   Ibrahim
# @Last modified time: 2019-12-12T13:55:38+02:00


import torch.nn as nn

class M3(nn.Module):
    def __init__(self):
        super(M3, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 80, 4)
        self.norm1 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(256, 256, 3)
        self.norm2 = nn.BatchNorm1d(256)
        self.avgpool = nn.AvgPool1d(498)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.norm1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(self.norm2(x))
        x = self.pool(x)
        x = self.avgpool(x)
        x = self.fc(x.view(-1,256))
        return x

class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.norm1 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.norm3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.norm4 = nn.BatchNorm1d(512)
        self.avgpool = nn.AvgPool1d(30)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.norm1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(self.norm2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(self.norm3(x))
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(self.norm4(x))
        x = self.pool(x)
        x = self.avgpool(x)
        x = self.fc(x.view(-1,512))
        return x
class M11(nn.Module):
    def __init__(self):
        super(M11, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.norm1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.norm2 = nn.BatchNorm1d(64)
        self.conv3_1 = nn.Conv1d(64, 128, 3)
        self.conv3_2 = nn.Conv1d(128, 128, 3)
        self.norm3 = nn.BatchNorm1d(128)
        self.conv4_1 = nn.Conv1d(128, 256, 3)
        self.conv4_2 = nn.Conv1d(256, 256, 3)
        self.norm4 = nn.BatchNorm1d(256)
        self.conv5_1 = nn.Conv1d(256, 512, 3)
        self.conv5_2 = nn.Conv1d(512, 512, 3)
        self.norm5 = nn.BatchNorm1d(512)
        self.avgpool = nn.AvgPool1d(25)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.norm1(x))
        x = self.pool(x)

        #2 X Conv2
        for i in range(2):
          x = self.conv2(x)
          x = self.relu(self.norm2(x))
        x = self.pool(x)

        #2 X Conv3
        # for i in range(2):
        x = self.conv3_1(x)
        x = self.relu(self.norm3(x))
        x = self.conv3_2(x)
        x = self.relu(self.norm3(x))
        x = self.pool(x)

        #3 X Conv4
        x = self.conv4_1(x)
        x = self.relu(self.norm4(x))
        for i in range(2):
          x = self.conv4_2(x)
          x = self.relu(self.norm4(x))
        x = self.pool(x)

        #2 X Conv5
        # for i in range(2):
        x = self.conv5_1(x)
        x = self.relu(self.norm5(x))
        x = self.conv5_2(x)
        x = self.relu(self.norm5(x))
        # x = self.pool(x)

        #Global Average Pooling
        x = self.avgpool(x)
        x = self.fc(x.view(-1, 512))
        return x

class M18(nn.Module):
    def __init__(self):
        super(M18, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.norm1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(4,2)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.norm2 = nn.BatchNorm1d(64)
        self.conv3_1 = nn.Conv1d(64, 128, 3)
        self.conv3_2 = nn.Conv1d(128, 128, 3)
        self.norm3 = nn.BatchNorm1d(128)
        self.conv4_1 = nn.Conv1d(128, 256, 3)
        self.conv4_2 = nn.Conv1d(256, 256, 3)
        self.norm4 = nn.BatchNorm1d(256)
        self.conv5_1 = nn.Conv1d(256, 512, 3)
        self.conv5_2 = nn.Conv1d(512, 512, 3)
        self.norm5 = nn.BatchNorm1d(512)
        self.avgpool = nn.AvgPool1d(240)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.norm1(x))
        x = self.pool(x)

        #4 X Conv2
        for i in range(4):
          x = self.conv2(x)
          x = self.relu(self.norm2(x))
        x = self.pool(x)

        #4 X Conv3
        x = self.conv3_1(x)
        x = self.relu(self.norm3(x))
        for i in range(3):
          x = self.conv3_2(x)
          x = self.relu(self.norm3(x))
        x = self.pool(x)

        #4 X Conv4
        x = self.conv4_1(x)
        x = self.relu(self.norm4(x))
        for i in range(2):
          x = self.conv4_2(x)
          x = self.relu(self.norm4(x))
        x = self.pool(x)

        #4 X Conv5
        x = self.conv5_1(x)
        x = self.relu(self.norm5(x))
        for i in range(3):
          x = self.conv5_2(x)
          x = self.relu(self.norm5(x))
        x = self.pool(x)
        x = self.avgpool(x)
        x = self.fc(x.view(-1,512))
        return x


class Residual(nn.Module):
    def __init__(self, in_channels, filter_size, n_filters):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.norm = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()


    def forward(self, x):
        original_input = x
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        original_input = torch.nn.functional.interpolate(x.unsqueeze(0), size=(x.shape[1], x.shape[2]))#original_input
        total = original_input.squeeze(0) + x
        out = self.relu(self.norm(total))
        return out


class M34_res(nn.Module):
    def __init__(self):
        super(M34_res, self).__init__()
        self.conv1 = nn.Conv1d(1, 48, 80, 4)
        self.res1 = Residual(48, 3, 48)
        self.res2 = Residual(96, 3, 96)
        self.res2_1 = Residual(48, 3, 96)
        self.res3 = Residual(192, 3, 192)
        self.res3_1 = Residual(96, 3, 192)
        self.res4 = Residual(384, 3, 384)
        self.res4_1 = Residual(192, 3, 384)
        self.pool = nn.MaxPool1d(4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.conv1(x)

        #3 X Res1
        x = self.res1(x)
        x = self.res1(x)
        x = self.res1(x)

        x = self.pool(x)

         #4 X Res2
        x = self.res2_1(x)
        x = self.res2(x)
        x = self.res2(x)
        x = self.res2(x)

        x = self.pool(x)

         #6 X Res3
        x = self.res3_1(x)
        x = self.res3(x)
        x = self.res3(x)
        x = self.res3(x)
        x = self.res3(x)
        x = self.res3(x)

        x = self.pool(x)

        #3 X Res4
        x = self.res4_1(x)
        x = self.res4(x)
        x = self.res4(x)

        x = self.avgpool(x)

        x = self.fc(x.view(-1,384))
        return x

def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
