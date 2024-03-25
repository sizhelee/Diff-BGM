import pdb
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rnn():

    # fetch options
    nlayers = 2
    idim = 512
    hdim = 256
    dropout = 0.5

    rnn = getattr(nn, "LSTM")(idim, hdim, nlayers,
                                batch_first=True, dropout=dropout,
                                bidirectional=True)
    return rnn


class VisualEncoder(nn.Module):
    """ RNN-based encoder network for sequence data
    """

    def __init__(self):
        super(VisualEncoder, self).__init__()

        # define layers --- word embedding and RNN
        emb_idim = 512
        self.emb_odim = 256
        self.embedding = nn.Embedding(emb_idim, self.emb_odim)
        self.rnn = get_rnn()  # == LSTM

    def make_src_mask(self, src):
        # src = [batch size, t, d]

        src_mask = (src != 0)[:,:,0]

        return src_mask

    def forward(self, visual_feats):
        """ encode query sequence using RNN and return logits over proposals
        Args:
            onehot: onehot vectors of query; [B, vocab_size]
            mask: mask for query; [B,L]
            out_type: output type [word-level | sentenve-level | both]
        Returns:
            w_feats: word-level features; [B,L,2*h]
            s_feats: sentence-level feature; [B,2*h]
        """
        visual_feats = visual_feats.clone()
        mask = self.make_src_mask(visual_feats)

        # encoding onehot data.
        max_len = visual_feats.size(1)  # == L
        length = mask.sum(1)  # [B,]
        pack_wemb = nn.utils.rnn.pack_padded_sequence(
            visual_feats, length.cpu(), batch_first=True, enforce_sorted=False)
        w_feats, _ = self.rnn(pack_wemb)
        w_feats, max_ = nn.utils.rnn.pad_packed_sequence(
            w_feats, batch_first=True, total_length=max_len)
        w_feats = w_feats.contiguous()  # [B,L,2*h]

        B, L, H = w_feats.size()
        idx = (length-1).long()  # 0-indexed
        idx = idx.view(B, 1, 1).expand(B, 1, H//2)
        fLSTM = w_feats[:, :, :H//2].gather(1, idx).view(B, H//2)
        bLSTM = w_feats[:, 0, H//2:].view(B, H//2)
        s_feats = torch.cat([fLSTM, bLSTM], dim=1).unsqueeze(1)

        return s_feats