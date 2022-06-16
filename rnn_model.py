import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# dataclass for parameters of encoder
@dataclass
class RNNParameters:
    cell_type : str = 'lstm'
    feat_dim : int = 3
    hidden_size : int = 128
    num_layers : int = 1
    droupout_fc : float = 0.1
    dropout_rnn : float = 0.0
    bidirectional : bool = False

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

@dataclass
class AttentionParameters:
    input_embed_dim : int
    source_embed_dim : int
    output_embed_dim : int
    bias : bool = False

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

@dataclass
class LinearParameters:
    in_dim : int
    out_dim : int
    n_hidden : int
    hidden_dim : int
    activation : str = "relu"

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

class LinearModel(nn.Module):
    def __init__(self, in_dim, n_hidden, hidden_dim, out_dim, activation="relu"):
        super(LinearModel, self).__init__()
        self.in_dim = in_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            self.activation = nn.ReLU

        if self.hidden_dim < 1:
            self.linear = nn.Linear(self.in_dim, self.out_dim)
        else:
            layers = [nn.Linear(self.in_dim, self.hidden_dim), self.activation()]
            for i in range(n_hidden-1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(self.activation())
            layers.append(nn.Linear(self.hidden_dim, self.out_dim))

            self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class RNNEncoder(nn.Module):
    """RNN encoder."""

    def __init__(
            self, cell_type='lstm', feat_dim=3, hidden_size=128, num_layers=1,
            dropout_fc=0.1, dropout_rnn=0.0, bidirectional=False,**kwargs):
        super(RNNEncoder, self).__init__()
        self.cell_type = cell_type.lower()
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.dropout_fc = dropout_fc
        self.dropout_rnn = dropout_rnn
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        input_size = feat_dim

        if self.cell_type == 'lstm':
            self.rnn = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=self.dropout_rnn)
        elif self.cell_type == 'gru':
            self.rnn = GRU(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=bidirectional,
                           dropout=self.dropout_rnn)
        else:
            self.rnn = RNN(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=bidirectional,
                           dropout=self.dropout_rnn)

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_seq):
        x = src_seq
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out, h_t = self.rnn(x)

        if self.cell_type == 'lstm':
            final_hiddens, final_cells = h_t
        else:
            final_hiddens, final_cells = h_t, None

        if self.dropout_fc>0:
            encoder_out = F.dropout(encoder_out, p=self.dropout_fc, training=self.training)

        if self.bidirectional:
            batch_size = src_seq.size(0)

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, batch_size, -1)

            final_hiddens = combine_bidir(final_hiddens)
            if self.cell_type == 'lstm':
                final_cells = combine_bidir(final_cells)

        #  T x B x C -> B x T x C
        encoder_out = encoder_out.transpose(0, 1)
        final_hiddens = final_hiddens.transpose(0, 1)
        if self.cell_type == 'lstm':
            final_cells = final_cells.transpose(0, 1)

        return encoder_out, (final_hiddens, final_cells)


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super(AttentionLayer, self).__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask=None):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        out = torch.cat((x, input), dim=1)
        x = F.tanh(self.output_proj(out))
        return x, attn_scores

def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def GRU(input_size, hidden_size, **kwargs):
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def RNN(input_size, hidden_size, **kwargs):
    m = nn.RNN(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m