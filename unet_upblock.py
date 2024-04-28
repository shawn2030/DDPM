import torch
import torch.nn as nn

# Upsample -> Resnet -> Attention 
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, upsample, numheads) -> None:
        super().__init__()

        self.upsample = upsample

        # this is the first convolutional layer which will have group norm, silu activation function and convolution layer
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

        # this is the time projection layer which should be concatenated with the output of first convolutional layer from resnet block
        # each resnet block will have one of these
        # the output of this should have the same number as out_channels as we will be concatenting it with output of convolutional layer
        # which has the output size equal to out_channels
        self.t_emb_dim_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)

        )

        # second resnet block
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        # attention part - normalization and then multihead attention
        self.attention_norm  = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=numheads, batch_first=True)

        # this can be used for the skip connection
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, 1)

        # if upsampling is true, then upsample using convolutiontranspose 2d with stride 2 
        # avg pooling can also be used or upsampling
        if self.upsample:
            self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)
        else:
            self.up_sample_conv = nn.Identity()


    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        out = x

        ####    RESNET BLOCK STARTS
        resnet_input = out

        # first resnet block
        out = self.residual_input_conv(out)

        # concatenating the time projection layer for given t_emb with the output of first convolutional layer
        # this makes the neural network understand/process the input image x as well as time embedding at the same time
        out = out + self.t_emb_dim_layers(t_emb)[:, :, None, None]          # converting the shape to [B, C, H, W]

        # second resnet block
        out = self.resnet_conv_second(out)

        # residual connection with resnet_input
        out  = out + self.residual_input_conv(resnet_input)

        ####    RESNET BLOCK ENDS...


        ###     ATTENTION BLOCK STARTS
        batch_size, channels, h, w = out.shape
        in_attention = out.reshape(batch_size, channels, h*w)           #### DONT KNOW WHY THIS IS DONE. NEED TO CHECK SELF-ATTENTION
        in_attention = self.attention_norm(in_attention)
        
        # to ensure the channel features are the last dimension
        in_attention = in_attention.transpose(1,2)  
        out_attention, _ = self.attention(in_attention, in_attention, in_attention)

        out_attention = out_attention.transpose(1,2).reshape(batch_size, channels, h, w)

        # output of resnet block is added to the time projection layer
        out = out + out_attention                    

        return out