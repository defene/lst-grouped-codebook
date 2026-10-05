"""Isolated model wiring; no patches to historical modules."""
from torch import nn
from eo_denoise.models import CrossViewModel
from eo_denoise.var_codec import VarCodec
from experiments.paper1.lst_residual.runtime import ResidualCodec, objective
from .quantizer import GroupedCodec


class ExperimentModel(CrossViewModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        assert config['data']['modalities'] == ['lst']
        assert config['residual_experiment']['predict_residual']
        codec = VarCodec if config['model']['groups'] == 1 else GroupedCodec
        self.codecs = nn.ModuleDict({'lst': ResidualCodec(codec(1, config['model']))})
