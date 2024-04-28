import torch
import torch.nn as nn
from unet_downblock import DownBlock
from unet_midblock import MidBlock
from unet_upblock import UpBlock


class UNet(nn.Module):
    def __init__(self, im_channels) -> None:
        super().__init__()
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.down_sample = [True, True, False]
        self.t_emb_dim = 128

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], 3, 1, 1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, 
                                        self.down_sample[i], numheads=4))

        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], t_emb_dim= self.t_emb_dim, numheads=4))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, upsample=self.down_sample[i], numheads=4))
            

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, 3, 1, 1)
        

    def get_time_embeddings(self, time_steps, t_emb_dim):
        """
        This method coverts the integer time space representation to embeddings using a fixed embedding space.
        This is the sinusoidal position embedding method sin(pos / 10000 ** ( 2i/d_model) )
        time steps is (B,1)
        time embeddings should be of size batch_size x time embeddings

        """
        factor = 10000 ** ((torch.arange(
            start=0, end=t_emb_dim//2, device=time_steps.device) / (t_emb_dim // 2)
        ))

        pos = time_steps[:, None].repeat(1, t_emb_dim // 2)
        t_emb =  pos / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)

        return t_emb
    

    def forward(self, x, t):
        out = self.conv_in(x)

        t_emb = self.get_time_embeddings(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []                                      # output in downblock is stored for skip connections with upblock
        for down in self.downs:
            # print(out.shape)
            down_outs.append(out)
            out = down(out, t_emb)                          # running the forward method of Downblock

        for mid in self.mids:
            # print(out.shape)
            out = mid(out, t_emb)                           # running the forward method of Midblock


        for up in self.ups:
            down_out = down_outs.pop()
            # print(out.shape, down_out.shape)
            out = up(out, down_out, t_emb)


        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out


        





