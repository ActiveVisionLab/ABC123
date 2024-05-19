import torch
import torch.nn as nn
import math
from einops import rearrange

class CountingHeadMultiLinearLayersSeperateSum(nn.Module):
    def __init__(
        self,
        feature_dim,
        channels=[1],
        resloution=28,
        padding_mode="zeros",
        linear_project_then_sum=False,
    ):
        super().__init__()
        self.num_counts = int(channels[0])
        num_ups = 4
        self.linear_project_then_sum = linear_project_then_sum
        self.count_feat_dim = int(2 ** num_ups)

        self.channels = channels
        modules = []
        self.net = torch.nn.Sequential()
        input = feature_dim
        output = int(self.num_counts * self.count_feat_dim)

        self.net.add_module(f"linear_0", nn.Linear(input, output))

        inputs = []
        outputs = []
        for i in range(num_ups):
            inputs.append(int(self.count_feat_dim / (2 ** i)))
            outputs.append(int(self.count_feat_dim / (2 ** (i + 1))))
        self.ups = torch.nn.Sequential()
        for i in range(num_ups):
            self.ups.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    inputs[i], outputs[i], 7, padding=3, padding_mode=padding_mode
                ),
            )

            if i != (num_ups - 1):
                self.ups.add_module(
                    f"relu_{i}", nn.ReLU(),
                )
                self.ups.add_module(
                    f"upsample_{i}", nn.UpsamplingBilinear2d(scale_factor=2)
                )

        if self.linear_project_then_sum:
            resloution_pixelwise = resloution * 2 ** (num_ups - 1)
            self.lp = torch.nn.Sequential()
            self.lp.add_module(
                f"linear_proj", nn.Linear(resloution_pixelwise ** 2, 1),
            )

    def forward(self, x):
        # n is the number of count predictions
        b, _c, h, w = x.shape
        x = rearrange(x, "b c h w -> (b h w) c")
        x = self.net(x)  
        x = rearrange(
            x,
            "(b h w) (n c) -> (b n) c h w",
            b=b,
            h=h,
            w=w,
            c=self.count_feat_dim,
            n=self.num_counts,
        )

        x_upsampled = self.ups(x)

        intermediate_image = x_upsampled.clone()

        intermediate_image = rearrange(
            intermediate_image, "(b n) 1 h w -> b n h w", b=b
        )
        x_upsampled = rearrange(x_upsampled, "(b n) 1 h w  -> b n (h w)", b=b)

        if self.linear_project_then_sum:
            x = self.lp(x_upsampled).squeeze()

        else:
            x = torch.sum(x_upsampled, dim=-1)

        return x, intermediate_image
