from torch import nn
from .observer import EMAPruneMinMaxObserver, EMAWeightedMSEObserver, MinMaxObserver, EMAMinMaxObserver, MSEObserver, \
    EMAMSEObserver, CollectObserver, EMAQuantileObserver, LSQPlusObserver
from .fake_quant import FixedFakeQuantize, LSQFakeQuantize, LSQPlusFakeQuantize
import torch.nn.functional as F
import copy
import torch

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'EMAMinMaxObserver':        EMAMinMaxObserver,        # More general choice.   # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'EMAMSEObserver':           EMAMSEObserver,     # noqa: E241
    'EMAQuantileObserver':      EMAQuantileObserver,
    'CollectObserver':          CollectObserver,  # noqa: E241
    'EMAWeightedMSEObserver':   EMAWeightedMSEObserver,
    'LSQPlusObserver':          LSQPlusObserver,
    'EMAPruneMinMaxObserver':   EMAPruneMinMaxObserver,
}

FakeQuantizeDict = {
    'FixedFakeQuantize':     FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241                       # noqa: E241
    'LSQFakeQuantize':       LSQFakeQuantize,        # learnable scale/zero_point  
    'LSQPlusFakeQuantize':   LSQPlusFakeQuantize,
}


class QuantizedModule(nn.Module):
    def __init__(self, backend='academic'):
        super().__init__()
        self.backend = backend


class QuantizedOperator():
    pass


class QConv2d(QuantizedOperator, nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups,
                 bias,
                 padding_mode,
                 w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight_fake_quant = FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 max_norm,
                 norm_type,
                 scale_grad_by_freq,
                 sparse,
                 _weight,
                 w_qconfig):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse,
                         _weight=_weight)
        self.weight_fake_quant = FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)

    def forward(self, input):
        return F.embedding(
            input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


module_type_to_quant_weight = {
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d,
    nn.Embedding: QEmbedding,
}


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )
    elif isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            )
    elif isinstance(module, nn.Embedding):
        return dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            _weight=None,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config):
    if module is None:
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, 'bias', None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
        # module.weight.data = WeightQuantizer(tmp, w_qconfig=config)
    return module


def ActivationQuantizer(a_qconfig):
    return FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit, symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)


