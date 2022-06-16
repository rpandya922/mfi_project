import torch
import torch.nn as nn
from collections import namedtuple

from rnn_model import RNNParameters, RNNEncoder, LinearParameters, LinearModel, AttentionParameters, AttentionLayer


class IntentionPredictor(nn.Module):
    def __init__(
            self, 
            hparams_rnn_hist : RNNParameters, 
            hparams_rnn_plan : RNNParameters, 
            hparams_attention : AttentionParameters,
            hparams_linear : LinearParameters
        ):
        super(IntentionPredictor, self).__init__()

        # RNN model for human and robot trajectory history
        self.hist_encoder = self._create_encoder(hparams_rnn_hist)
        # RNN model for robot's future plan
        self.plan_encoder = self._create_encoder(hparams_rnn_plan)
        # attention layer for history encoder
        # TODO: remove AttentionLayer for now (until I can understand how to use it better)
        self.hist_attn = self._create_attention(hparams_attention)
        # linear layers for classifier
        self.classifier = self._create_classifier(hparams_linear)

    def _create_encoder(self, hparams):
        encoder = RNNEncoder(**hparams)
        return encoder

    def _create_attention(self, hparams):
        attention = AttentionLayer(**hparams)
        return attention

    def _create_classifier(self, hparams):
        linear = LinearModel(**hparams)
        return linear

    def forward(self, seq_hist, r_plan, goals):
        """
        sec : state sequence
        goals : the possible goals of the human
        """
        
        hist_enc_out, (hist_enc_hidd, _) = self.hist_encoder(seq_hist)
        plan_enc_out = self.plan_encoder(r_plan)[0]

        import ipdb; ipdb.set_trace()

def create_model():
    # create RNNParameters object
    rnn_hist_params = RNNParameters(
                    feat_dim=8,
                    num_layers=2,
                    hidden_size=128,
                    droupout_fc=0.0
                    )

    rnn_plan_params = RNNParameters(
                    feat_dim=4,
                    num_layers=2,
                    hidden_size=128,
                    droupout_fc=0.0
                    )

    # create AttentionParameters object
    attn_params = AttentionParameters(
                    input_embed_dim=128,
                    source_embed_dim=128,
                    output_embed_dim=32
                    )

    # TODO: determine in_dim for linear layers based on 
    # create LinearParameters object
    fc_params = LinearParameters(
                    in_dim=12,
                    out_dim=3,
                    n_hidden=2,
                    hidden_dim=128
                    )

    model = IntentionPredictor(rnn_hist_params, rnn_plan_params, attn_params, fc_params)

    return model