import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, CUDA, classes_num, pretrained_ViT=True, freeze_ViT=False):
        super(ViT, self).__init__()

        self.spatial_feat_dim = 32
        self.num_classes = classes_num
        self.nhid = 128
        self.levels = 8
        self.kernel_size = 3
        self.dropout = .1
        self.channel_sizes = [self.nhid] * self.levels
        self.CUDA = CUDA

        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained_ViT)
        num_ftrs = 1000
        self.linear_fc_layer = nn.Linear(num_ftrs, self.classes_num)

        # if freeze_resnet is True, requires grad is False
        for param in self.model.parameters():
            param.requires_grad = not freeze_ViT

    def forward(self, data):
        output = self.model(data)
        output = self.linear_fc_layer(output)
        return output
