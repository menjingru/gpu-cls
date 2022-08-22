import torch.nn as nn
from gpu_cls.model.skvnet_def122 import res_block

#
class SKVNet(nn.Module):
    def __init__(self,num_classes=2):  # hcc、fhcc
        super(SKVNet, self).__init__()
        # self.layer0 = spp_pool(1,8)  # 1,2,4,8,16,32,64,128
        self.layer1 = res_block(1, 16, "forward1")  # 128 × 6                        32
        # self.layer1 = res_block(1, 16, "forward1")  # 128 × 6                        32
        self.layer1down = res_block(16,16,"deconv")  # 64 × 16                       16
        self.layer2 = res_block(16, 32, "forward1")  # 64 × 32                       16
        self.layer2down = res_block(32,32,"deconv")  # 32 × 32                       8
        self.layer3 = res_block(32, 64, "forward1")  # 32 × 64                       8
        self.layer3down = res_block(64,64,"deconv")  # 16 × 64
        self.layer4 = res_block(64, 128, "forward1")  # 16 × 128
        # self.layer4down = res_block(128,128,"deconv")  # 8 × 128
        self.layer5 = res_block(128, 64, "forward1")  # 8 × 64
        self.layer5down = res_block(64,64,"deconv")  # 4 × 64
        self.layer6 = res_block(64, 32, "forward1")  # 4 × 32
        self.layer6down = res_block(32,32,"deconv")  # 2 × 32
        self.layer7 = res_block(32, 16, "forward1")  # 2 × 16
        self.layer7down = res_block(16,16,"deconv")  # 1 × 16
        self.layer10 = res_block(16, num_classes, "pointconv")  ###   num_classes=2
        # self.softmax = f.softmax
        # self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
        # self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化
        # self.fc0 = nn.Linear(4, 4)  # .cuda()
        self.prelu = nn.ELU()

        ####  提取特征

    def forward(self,x):
        # with torch.no_grad():
        x = self.layer1(x)
        x = self.layer1down(x)
        # print(x.shape)

        x = self.layer2(x)
        x = self.layer2down(x)
        # print(x.shape)

        x = self.layer3(x)
        x = self.layer3down(x)
        # print(x.shape)

        x = self.layer4(x)
        # x = self.layer4down(x)
        # print(x.shape)

        x = self.layer5(x)
        x = self.layer5down(x)
        # print(x.shape)

        x = self.layer6(x)
        x = self.layer6down(x)
        # print(x.shape)

        x = self.layer7(x)
        x = self.layer7down(x)
        # print(x.shape)

        out = self.layer10(x)
        # out = self.softmax(out,dim=1)

        return out


# model = SKVNet(2)
