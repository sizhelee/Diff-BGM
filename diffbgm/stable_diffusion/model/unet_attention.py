"""
---
title: Transformer for Stable Diffusion U-Net
summary: >
 Annotated PyTorch implementation/tutorial of the transformer
 for U-Net in stable diffusion.
---

# Transformer for Stable Diffusion [U-Net](unet.html)

This implements the transformer module used in [U-Net](unet.html) that
 gives $\epsilon_\text{cond}(x_t, c)$

We have kept to the model definition and naming unchanged from
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
so that we can load the checkpoints directly.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding
)


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                ) for _ in range(n_layers)
            ]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, height, width]`
        b, c, h, w = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross attention layer and pre-norm layer
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        # Feed-forward network
        x = self.ff(self.norm3(x)) + x
        #
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        w_size: int = 2,
        is_inplace: bool = True
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        self.w_size = w_size

        # Attention scaling factor
        self.scale = d_head**-0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat(
                (qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1
            )

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, : self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[: 2], self.n_heads, -1)
        k = k.view(*k.shape[: 2], self.n_heads, -1)
        v = v.view(*v.shape[: 2], self.n_heads, -1)

        seg_len = q.shape[1]
        mask_one = torch.zeros(q.shape[1], k.shape[1])
        for i in range(self.w_size):
            mask_one[seg_len*i//self.w_size:seg_len*(i+1)//self.w_size, k.shape[1]*i//self.w_size:k.shape[1]*(i+1)//self.w_size] = 1
        mask = mask_one.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(q.shape[0], q.shape[2], seg_len, k.shape[1]).cuda()
        mask = (mask==1)
        # mask = torch.stack([mask_one for i in range(q.shape[0]*q.shape[2])]) == 1
        # mask = mask.reshape(q.shape[0], q.shape[2], seg_len, k.shape[1]).cuda()
        mask_num = -2**31 * torch.ones_like(mask).half()

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        attn = torch.where(mask, attn, mask_num)

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half :] = attn[half :].clone().softmax(dim=-1)
            attn[: half] = attn[: half].clone().softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[: 2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, video_attention_index, audio_attention_index, frame_size, audio_per_frame):
        """
        Apply QKV attention.
        : attention_index_v:[V_len x H]
        : attention_index_a:[A_len, H]
        :param qkv: an [ N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
    
        
        bs, width, _ = qkv.shape
        video_len = video_attention_index.shape[0] 
        audio_len = audio_attention_index.shape[0]
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        v_as = []
        a_as = []
        video_q = q[:, :, :video_len] #[bsz, c*head, videolength]
        audio_q = q[:, :, video_len:] 
      
        for idx in range(0, video_len//frame_size):
            video_frame_k = th.index_select(k, -1, video_attention_index[idx*frame_size]) #[bsz, c*head, k_num]
            video_frame_v = th.index_select(v, -1, video_attention_index[idx*frame_size]) #[bsz, c*head, k_num]
            video_frame_q = video_q[:, :, idx*frame_size:(idx+1) * frame_size]
            
            w_slice =  th.einsum(
            "bct,bcs->bts",
            (video_frame_q * scale).view(bs * self.n_heads, ch, -1),
            (video_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards
            
            w_slice = th.softmax(w_slice, dim=-1) #[bsz, 1, k_len]
            a = th.einsum("bts,bcs->bct", w_slice, video_frame_v.view(bs * self.n_heads, ch, -1)).reshape(bs * self.n_heads, ch, -1)
            v_as.append(a)

            audio_frame_k = th.index_select(k, -1, audio_attention_index[idx*audio_per_frame])#[bsz, c*head, k_num]
            audio_frame_v = th.index_select(v, -1, audio_attention_index[idx*audio_per_frame]) #[bsz, c*head, k_num]
            if idx == (video_len//frame_size-1): 
                audio_frame_q = audio_q[:, :, idx*audio_per_frame:]
            else:
                audio_frame_q = audio_q[:, :, idx*audio_per_frame:(idx +1)* audio_per_frame]
            w_slice =  th.einsum(
            "bct,bcs->bts",
            (audio_frame_q * scale).view(bs * self.n_heads, ch, -1),
            (audio_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1) #[bsz, 1, k_len]
            a = th.einsum("bts,bcs->bct", w_slice, audio_frame_v.view(bs * self.n_heads, ch, -1)).reshape(bs * self.n_heads, ch, -1)
            a_as.append(a)
      
        v_a = th.cat(v_as, dim=2)
        a_a = th.cat(a_as, dim=2)

        return v_a.reshape(bs, -1,  video_len), a_a.reshape(bs, -1, audio_len)


class CrossAttentionBlock(nn.Module):
    """
    RS-MMA: ramdom based multi-modal attention block
    An attention block that allows cross attention .
    :param local_window: local window size.
    :param window_shift: whether to random shift the window.
    """

    def __init__(
        self,
        channels,  
        num_heads=1,
        local_window = 1,
        window_shift = False,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = num_heads
        self.local_window = local_window
        self.window_shift = window_shift
        self.v_norm = normalization(self.channels)
        self.a_norm = normalization(self.channels)
        self.v_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.a_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.va_index=None
        self.av_index=None
 
    def attention_index(self, audio_size, video_size, device):
        f, h, w = video_size
        l = audio_size
        video_len = f * h * w
        audio_len_perf = int(l / f)
        if self.window_shift:
            window_shift = random.randint(0, f-self.local_window)
        else:
            window_shift = 0
        
        if self.va_index == None:
            va_index_x =  th.arange(0,  self.local_window*audio_len_perf).view(1, -1) 
            va_index_y = th.arange(0, f).unsqueeze(-1).repeat(1, h*w).view(-1, 1)
            va_index_y = va_index_y * audio_len_perf
            self.va_index = (va_index_y + va_index_x).to(device)

        va_index = (self.va_index +audio_len_perf*window_shift) %l + video_len

        if self.av_index == None:
            av_index_x = th.arange(0,  self.local_window*h*w).view(1, -1)
            av_index_y = th.arange(0, f).unsqueeze(-1).repeat(1, audio_len_perf).view(-1, 1)
            av_index_y = av_index_y * h*w
            self.av_index =  (av_index_y + av_index_x).to(device)

        av_index = (self.av_index + h*w*window_shift)%video_len
        
        attention_index_v = va_index
        attention_index_a = av_index
        
        # complete attention index
        if attention_index_a.shape[0] < l:
            attention_index_a = th.cat([attention_index_a, attention_index_a[-1*(l-attention_index_a.shape[0]):]], dim=0)
    
        return attention_index_v, attention_index_a
        

    def forward(self, audio, time_step, video): 
        b, f, c, h, w = video.shape
        b, c, l = audio.shape
        
        video_token = rearrange(video, 'b f c h w -> b c (f h w)')
        audio_token = audio

        attention_index_v, attention_index_a = self.attention_index((l), (f, h, w), video.device)

        v_qkv = self.v_qkv(self.v_norm(video_token)) #[bsz, c, f*h*w+l]
        a_qkv = self.a_qkv(self.a_norm(audio_token)) #[bsz, c, f*h*w+l]
        qkv = th.concat([v_qkv, a_qkv], dim=2)

        video_h, audio_h = self.attention(qkv, attention_index_v, attention_index_a, h*w, int(l/f))
        video_h = rearrange(video_h, 'b c (f h w)-> b f c h w ', f=f, h=h)

        video_h = video + video_h
        audio_h = audio + audio_h
      
        
        return video_h, audio_h


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """
    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult), nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)
