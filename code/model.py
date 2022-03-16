from resnet import resnet50
import torch.nn as nn
import torch
from transformers import AutoModel


class ResNet(nn.Module):
    def __init__(self, num_character):
        super().__init__()
        self.num_character = num_character

        # Align vocab_size and num_cls
        self.LM = AutoModel.from_pretrained('bert-base-chinese', requires_grad=False)
        vocab_size = self.LM.config.vocab_size

        # Initialize the vision model
        self.VM = resnet50(pretrained=True, num_classes=vocab_size)

    def forward(self, x):
        """
        x: (B, H, W)
        """
        x = torch.cat(torch.chunk(x, self.num_character, dim=2), dim=0)  # (num_character*B, H, W/num_character)
        y = self.VM(x)                                                   # (num_character*B, vocab_size)
        return y


if __name__ == '__main__':
    batch = torch.randn(4, 3, 40, 80)
    model = ResNet(num_character=2)
    out = model(batch)
