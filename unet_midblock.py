import torch
import torch.nn as nn


# Resnet block -> self attention block -> resnet block
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, numheads) -> None:
        super().__init__()

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            ),

            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            )
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ), 

            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels),
                # nn.Linear(out_channels, out_channels)
            )
        ])

        self.resnet_conv_second = nn.ModuleList([
             nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            ),

            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            )
        ])

        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=numheads, batch_first=True)

        self.resnet_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        ])


    def forward(self, x, t_emb):
        out = x

        #####   FIRST RESNET BLOCK STARTS ...
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        place_holder = self.resnet_input_conv[0](resnet_input)
        out = out + place_holder

        #####   FIRST RESNET BLOCK ENDS ...

        ####    ATTENTION BLOCK STARTS ...
        batch_size, channels, h, w = out.shape
        in_attention = out.reshape(batch_size, channels, h*w)           #### DONT KNOW WHY THIS IS DONE. NEED TO CHECK SELF-ATTENTION
        in_attention = self.attention_norm(in_attention)
        
        # to ensure the channel features are the last dimension
        in_attention = in_attention.transpose(1,2)  
        out_attention, _ = self.attention(in_attention, in_attention, in_attention)

        out_attention = out_attention.transpose(1,2).reshape(batch_size, channels, h, w)

        # output of resnet block is added to the time projection layer
        out = out + out_attention 

         ####    ATTENTION BLOCK ENDS ...

        #####   SECOND RESNET BLOCK STARTS ...
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)

        out = out + self.resnet_input_conv[1](resnet_input)

        #####   SECOND RESNET BLOCK ENDS ...

        return out  