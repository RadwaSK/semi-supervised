import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, CUDA, classes_num, pretrained_ViT=True, freeze_ViT=False):
        super(ViT, self).__init__()

        self.num_classes = classes_num
        self.CUDA = CUDA

        self.model = timm.create_model("vit_base_patch32_224", pretrained=pretrained_ViT)
        self.linear_fc_layer = nn.Linear(1000, self.num_classes)

        # if freeze_resnet is True, requires grad is False
        for param in self.model.parameters():
            param.requires_grad = not freeze_ViT

    def forward(self, data):
        output = self.model(data)
        output = self.linear_fc_layer(output)
        return output
