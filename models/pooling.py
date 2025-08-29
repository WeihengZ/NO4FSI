import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

'''
U-Net model
'''
# Helper functions
def pad_to_even(x):
    _, _, M, N = x.shape
    pad_h = 1 if M % 2 != 0 else 0
    pad_w = 1 if N % 2 != 0 else 0
    out = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    return out

def remove_padding(x, target_shape):
    '''
    x: (B, F, M, N)

    return (B, T, F, M, N)
    '''
    B, _, M, N = x.shape
    _, _, M_new, N_new = target_shape
    if M > M_new:
        x = x[:,:,:M_new,:]
    if N > N_new:
        x = x[:,:,:,:N_new]

    return x

# Double convolution layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv(x)

# Down-sample layers
class conditioned_down(nn.Module):
    def __init__(self):
        super(conditioned_down, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        '''
        Two possible input size:
            1) (B, F, M, N)
            2) (B, T, F, M, N)
        return:
            1) (B, F, M, N)
            2) (B, T, F, M, N)
        '''
        if len(x.shape) == 4:
            x = pad_to_even(x)
            x = self.pool(x)
        else:
            B, T, F_, M, N = x.shape
            x = x.reshape(-1, F_, M, N)    # (BT, F, M, N)
            x = pad_to_even(x)    # (BT, F, M, N)
            x = self.pool(x)    # (BT, F, M/2, N/2)
            BT, F_, M, N = x.shape
            x = x.reshape(B, T, F_, M, N)
        return x

# Up-sample layers
class conditioned_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conditioned_up, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, target_shape):
        '''
        Input shape: (B, T, F, M/2, N/2)
        Output shape: (B, T, F, M', N')
        '''
        # up sampling on the original map
        B, F_, M, N = x.shape
        x = self.conv_transpose(x.reshape(-1, F_, x.shape[-2], x.shape[-1]))    # (BT, F, M, N)
        x = x.reshape(B, F_, x.shape[-2], x.shape[-1])
        
        # extract the target shape
        x = remove_padding(x, target_shape)    # (B, F, M, N)

        # reshape
        x = x.reshape(B, F_, x.shape[-2], x.shape[-1])    # (B, T, F, M, N)

        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, width):
        super(UNet2D, self).__init__()

        self.enc1 = DoubleConv(in_channels, width)
        self.enc2 = DoubleConv(width, width)
        self.down1 = conditioned_down()
        self.down2 = conditioned_down()

        self.enc3 = DoubleConv(width, width)
        self.enc4 = DoubleConv(2 * width, width)
        self.enc5 = DoubleConv(2 * width, width)
        self.up1 = conditioned_up(width, width)
        self.up2 = conditioned_up(width, width)

        self.out_map = nn.Linear(width, out_channels)
    
    def forward(self, x):
        '''
        x: (B, F, M, N)
        '''

        # encoding
        e1 = self.enc1(x)    # (B, F, M, N)
        e2 = self.down1(e1)    # (B, F, M/2, N/2)
        e2 = self.enc2(e2)     # (B, F, M/2, N/2)
        e3 = self.down2(e2)    # (B, F, M/4, N/4)

        # decoding
        d3 = self.enc3(e3)     # (B, F, M/4, N/4)
        d2 = torch.cat((self.up1(d3, e2.shape), e2), 1)    # (B, 2F, M/2, N/2)
        d2 = self.enc4(d2)    # (B, F, M/2, N/2)
        d1 = torch.cat((self.up2(d2, e1.shape), e1), 1)    # (B, 2F, M, N)
        d0 = self.enc5(d1)    # (B, F, M, N)
        
        # reshape
        out = d0.permute(0,2,3,1)    # (B, M, N, F)
        out = self.out_map(out)    # (B, M, N, F)

        return out

'''
FNO layer
'''
# simple FNO blocks
def compl_mul2d(a, b):
    # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 0], b[..., 1]) + op(a[..., 1], b[..., 0])
    ], dim=-1)

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        real_part = x_ft.real
        imag_part = x_ft.imag
        x_ft = torch.stack([real_part, imag_part], dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Perform complex multiplication
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        # Return to physical space
        x = torch.fft.irfftn(out_ft, dim=(-2, -1), norm="ortho", s=(x.size(-2), x.size(-1)))
        return x

'''
Vision transformer
'''
class PatchEmbedding(nn.Module):
    """
    Splits image into patches and projects them into a higher-dimensional space.
    """
    def __init__(self, img_size=298, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x: [batch_size, 3, 298, 298]
        x = self.proj(x)  # [batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5]
        x = x.flatten(2)  # Flatten to [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # Transpose to [batch_size, n_patches, embed_dim]
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=298, patch_size=16, in_channels=3, out_channels=3, embed_dim=768, num_heads=8, num_layers=6):
        super(VisionTransformer, self).__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))

        # Output projection layer to match output dimension [pred_len, 3, 298, 298]
        self.out_channels = out_channels
        self.fc = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        self.vel_compressor = nn.Sequential(nn.Linear(2,16), nn.Tanh(), nn.Linear(16,1))

    def forward(self, x):
        # Input x: [batch_size, 298, 298, 3]
        x = x.permute(0, 3, 1, 2)  # [batch_size, 3, 298, 298]

        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]

        # Concatenate class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, n_patches + 1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.transformer(x)  # [batch_size, n_patches + 1, embed_dim]

        # Take cls_token output
        cls_token_final = x[:, 0]  # [batch_size, embed_dim]

        # Output projection
        x = self.fc(cls_token_final)  # [batch_size, pred_len * 3 * patch_size * patch_size]

        # Reshape to match output dimensions
        patch_size = self.patch_embed.patch_size
        x = x.view(batch_size, self.out_channels, patch_size, patch_size)  # [batch_size, F, patch_size, patch_size]

        # Upsample to original image size
        x = torch.nn.functional.interpolate(
            x.view(batch_size, self.out_channels, patch_size, patch_size),
            size=(self.patch_embed.img_size, self.patch_embed.img_size),
            mode="bilinear"
        )

        x = x.permute(0, 2, 3, 1)  # [batch_size, 298, 298, 3]

        return x

'''
Iterative model
'''
class Pooling_model(nn.Module):
    def __init__(self, modes_x, modes_y, width, backend):
        super(Pooling_model, self).__init__()

        self.backend = backend
        self.uvp_map = nn.Linear(3, width)
        self.bc_map = nn.Linear(2, width)

        #
        self.compress_map = nn.Linear(2*width, width)
        self.fno = FourierLayer(in_channels=width, out_channels=width, modes1=modes_x, modes2=modes_y)
        self.Unet = UNet2D(in_channels=width, out_channels=width, width=width)
        self.VT = VisionTransformer(in_channels=width, out_channels=width)

        #
        self.out_map = nn.Linear(width, 3)
    
    def forward(self, x, bc_map, xyt):
        '''
        Inputs:
            x (B, M, N, F)
            bc_map (B, T, M, N, F)
        return:
            outputs: (B, T, M, N, 3)
        '''
        B, M, N, _ = x.shape
        x = self.uvp_map(x)
        bc_map = self.bc_map(bc_map)
        x_bc_map = torch.cat((x, torch.mean(bc_map,1)), -1)    # (B, M, N, 2*F)

        if self.backend == 'FNO':
            x_bc_map = self.compress_map(x_bc_map)
            out = self.fno(x_bc_map.permute(0,3,1,2)).permute(0,2,3,1)
            return self.out_map(out)

        if self.backend == 'Unet':
            x_bc_map = self.compress_map(x_bc_map)
            out = self.Unet(x_bc_map.permute(0,3,1,2))
            return self.out_map(out)
        
        if self.backend == 'VT':
            x_bc_map = self.compress_map(x_bc_map)
            out = self.VT(x_bc_map)
            return self.out_map(out)

        
