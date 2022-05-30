import numpy as np   # noqa: F401
import torch
import torch.nn as nn
import os
import seaborn as sns
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from .util_quant import fake_quantize_per_tensor_affine
color1 = 'steelblue'
color2 = 'firebrick'
color3 = 'darkcyan'
color4 = 'darkorange'


def quantile_range(x, percentile):
    upper = torch.quantile(x.abs(), percentile)
    return -upper, upper


class ObserverBase(nn.Module):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_name(self, name):
        self.name = name

    def set_batch(self, batch):
        self.batch = batch

    def set_percentile(self, percentile):
        self.percentile = percentile

    def cac_thres(self, token_min, token_max):
        # token_min = self.norm_vector(token_min)
        # token_max = self.norm_vector(token_max)
        _, upper = quantile_range(token_max, self.percentile)
        lower, _ = quantile_range(token_min, self.percentile)
        indice_upper = torch.nonzero(token_max <= upper, as_tuple=True)[0]
        indice_lower = torch.nonzero(token_min >= lower, as_tuple=True)[0]
        return indice_lower, indice_upper

    def prune_token(self, value):   # try batch first
        if 'attention_probs' in self.name:
            return value
        token_max, _ = value.max(1)
        token_min, _ = value.min(1)
        indice_lower, indice_upper = self.cac_thres(token_min, token_max)
        upper = token_max[indice_upper].max()
        lower = token_min[indice_lower].min()
        value = torch.clip(value, max=upper, min=lower)
        return value

    def remove_padding(self, x, observation_mask, seq_pos):
        # assert the first dim is batch
        pos = list(range(len(x.shape)))
        shape = x.shape
        pos.remove(seq_pos)
        if len(pos) == 3:
            x = x.permute(pos[0], seq_pos, pos[1], pos[2]).reshape(shape[pos[0]], shape[seq_pos], -1)
        if len(pos) == 2:
            x = x.permute(pos[0], seq_pos, pos[1])
        # print(x.shape)
        value = torch.Tensor().cuda()
        # print('the name is {}'.format(self.name), flush=True)
        for sum, seq in zip(observation_mask, x):
            value = torch.cat((value, seq[: sum]), 0)
        return value

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class CollectObserver(ObserverBase):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(CollectObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def collect_values(self, name, log, draw_type=1, seq=None):
        # type=1 means draw distribution
        # type=2 means draw min and max per-token
        self.name = name
        self.log = log
        if not os.path.exists(self.log):
            os.makedirs(self.log)
        self.draw_type = draw_type
        self.seq = seq  # list to represent for many tokens

    def draw_distribution(self, x_orig):
        tmp = x_orig.reshape(-1).cpu().numpy()
        print('the collector is {}, the shape is {}'.format(self.name, x_orig.reshape(-1).shape))
        print('the min and max is {}, {}'.format(tmp.min(), tmp.max()))
        plt.figure(figsize=(30, 10), dpi=300)
        plt.xlim(tmp.min(), tmp.max())
        plt.hist(tmp, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("input")
        # 显示纵轴标签
        plt.ylabel("nums")
        # 显示图标题
        plt.title("act distribution")
        plt.savefig(os.path.join(self.log, self.name))
        plt.close()

    def draw_token_distribution(self, x_orig):
        if len(x_orig.shape) == 2:
            return
        tmp = x_orig[0]
        seq_len = len(self.seq)
        token_min, token_min_pos = list(tmp.min(1))
        token_max, token_max_pos = list(tmp.max(1))
        token_min = token_min.cpu().numpy()
        token_max = token_max.cpu().numpy()
        plt.figure(figsize=(30, 10), dpi=300)
        plt.plot(np.arange(seq_len), token_min, color=color1)
        plt.plot(np.arange(seq_len), token_max, color=color2)
        plt.xticks(np.arange(seq_len), self.seq, rotation=30)
        plt.savefig(os.path.join(self.log, self.name))
        plt.close()

    def draw_heatmap(self, x_orig):
        if len(x_orig.shape) == 2:
            return
        plt.figure(figsize=(120, 40))
        sns.heatmap(x_orig[0].cpu().numpy(), yticklabels=self.seq)
        plt.savefig(os.path.join(self.log, self.name))
        plt.close()

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        if observation_mask is not None:
            x_orig = self.remove_padding(x_orig, observation_mask, seq_pos)
        if self.draw_type == 1:
            self.draw_distribution(x_orig)
        if self.draw_type == 2:
            self.draw_token_distribution(x_orig)
        if self.draw_type == 3:
            self.draw_heatmap(x_orig)


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x


class LSQPlusObserver(ObserverBase):
    '''
    LSQ+ observer. This only suits for weight Observer
    '''

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(LSQPlusObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        assert self.symmetric is True
        self.mean = None
        self.std = None

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.mean = y.mean(1)
            self.std = y.std(1)
        self.min_val = self.mean - 3 * self.std
        self.max_val = self.mean + 3 * self.std
        return x


class EMAMinMaxObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1, ema_ratio=0.9):
        super(EMAMinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.ema_ratio = ema_ratio
        self.cnt = 0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt
        return x


class EMAPruneMinMaxObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1, ema_ratio=0.9):
        super(EMAPruneMinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.ema_ratio = ema_ratio
        self.cnt = 0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
            x = self.prune_token(x)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt
        return x


class EMAQuantileObserver(ObserverBase):
    """Moving average quantile among batches.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1, ema_ratio=0.9,
                 threshold=0.99999, bins=2048):
        super(EMAQuantileObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        assert self.ch_axis == -1, "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0., max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(min_val_cur, -clip_value) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(max_val_cur, clip_value) * (1.0 - self.ema_ratio)
        return x


class MSEObserver(ObserverBase):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1, p=2.0, num=100):
        super(MSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = p
        self.num = num  # candidate num
        self.opt_method = 'golden_selection'
        assert self.symmetric is False

    def lp_loss(self, pred, tgt, p=2.0):
        """
        loss function measured in L_p Norm
        """
        return (pred - tgt).abs().pow(p).mean()

    def loss_fx(self, x, new_min, new_max):
        new_min = torch.tensor(new_min).cuda()
        new_max = torch.tensor(new_max).cuda()
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = fake_quantize_per_tensor_affine(
                    x, scale.item(), int(zero_point.item()),
                    self.quant_min, self.quant_max)
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_2D_search(self, x):
        if self.ch_axis != -1:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / float(self.quant_max - self.quant_min)
            # enumerate zp
            for zp in range(self.quant_min, self.quant_max + 1):
                new_min = max(tmp_min - zp * tmp_delta, x_min)
                new_max = min(tmp_max - zp * tmp_delta, x_max)
                score = self.loss_fx(x, new_min, new_max)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = min(score, best_score)
        return best_min, best_max

    def golden_asym_shift_loss(self, shift, xrange, x, x_min, x_max):
        tmp_min = 0.0
        tmp_max = xrange
        new_min = max(tmp_min - shift, x_min)
        new_max = min(tmp_max - shift, x_max)
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_asym_range_loss(self, xrange, x, x_min, x_max):
        tmp_delta = xrange / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(xrange, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method='Bounded',
        )
        return result.fun

    def golden_section_search_channel(self, x, x_min, x_max):
        xrange = x_max - x_min
        result = minimize_scalar(
            self.golden_asym_range_loss,
            args=(x, x_min, x_max),
            bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
            method='Bounded',
        )
        final_range = result.x
        tmp_min = 0.0
        tmp_max = final_range
        tmp_delta = final_range / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        subresult = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(final_range, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method='Bounded',
        )
        final_shift = subresult.x
        best_min = max(tmp_min - final_shift, x_min)
        best_max = min(tmp_max - final_shift, x_max)
        return torch.tensor(best_min).cuda(), torch.tensor(best_max).cuda()

    def golden_section_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_channel(x, x_min, x_max)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(x_channel):
                x_min[ch], x_max[ch] = self.golden_section_search_channel(
                    x_channel[ch], x_min[ch], x_max[ch])
        return x_min, x_max

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.opt_method == 'grid':
            best_min, best_max = self.perform_2D_search(x)
        else:
            best_min, best_max = self.golden_section_search(x)
        self.min_val = best_min
        self.max_val = best_max
        return x


class EMAMSEObserver(MSEObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1, p=2.0, num=100, ema_ratio=0.9):
        super(EMAMSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis, p=p, num=num)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.opt_method == 'grid':
            best_min, best_max = self.perform_2D_search(x)
        else:
            best_min, best_max = self.golden_section_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.ema_ratio + best_min * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + best_max * (1.0 - self.ema_ratio)
        return x


class EMAWeightedMSEObserver(EMAMSEObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1, p=2.0, num=100, ema_ratio=0.9):
        super(EMAWeightedMSEObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis,
            p=p, num=num, ema_ratio=ema_ratio
            )
        self.grad = None

    def set_grad(self, grad):
        self.grad = grad.clone().detach()

    def lp_loss(self, pred, tgt, p=2.0):
        """
        loss function measured in L_p Norm
        """
        return ((pred - tgt) * self.grad).pow(p).sum()

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
            if self.grad is not None:
                self.grad = self.remove_padding(self.grad, observation_mask, seq_pos)
                self.grad = self.grad / self.grad.abs().sum()
        if self.opt_method == 'grid':
            best_min, best_max = self.perform_2D_search(x)
        else:
            best_min, best_max = self.golden_section_search(x)
        self.grad = None
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.ema_ratio + best_min * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + best_max * (1.0 - self.ema_ratio)
        return x
