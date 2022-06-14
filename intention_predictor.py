import torch
import torch.nn as nn

from rnn_model import RNNParameters, RNNEncoder, 
                      LinearParameters, LinearModel, 
                      AttentionParameters, AttentionLayer


class IntentionPredictor(nn.Module):
    def __init__(self, hparams_rnn : RNNParameters, hparams_linear : LinearParameters, 
                 hparams_attention : AttentionParameters):
        super(IntentionPredictor, self).__init__()

        self.encoder = self._create_encoder(hparams_rnn)
        self.attention = self._create_attention(hparams_attention)
        self.classifier = self._create_classifier(hparams_linear)

    def _create_encoder(self, hparams):
        encoder = RNNEncoder(**hparams)
        return encoder

    def _create_attention(self, hparams):
        attention = AttentionLayer(hparams)
        return attention

    def _create_classifier(self, hparams):
        linear = LinearModel(hparams)
        return linear

    def forward(self, x):
        # TODO: make architecture
        pass

def create_model():
    # TODO: create RNNParameters object

    # TODO: create LinearParameters object

    # TODO: create 