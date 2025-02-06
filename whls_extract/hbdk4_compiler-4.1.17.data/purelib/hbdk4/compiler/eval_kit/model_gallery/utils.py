import torch


def _dfs(x, type, func):
    if isinstance(x, (list, tuple)):
        return [_dfs(i, type, func) for i in x]
    if isinstance(x, dict):
        return {k: _dfs(x[k], type, func) for k in sorted(x.keys())}
    if isinstance(x, type):
        return func(x)
    return x


def _flatten(x):
    ret = []
    if isinstance(x, (list, tuple)):
        for i in x:
            ret.extend(_flatten(i))
    elif isinstance(x, dict):
        for k in sorted(x.keys()):
            ret.extend(_flatten(x[k]))
    else:
        ret = [x]
    return ret


def nhwc_nchw(x):
    return _dfs(x, torch.Tensor, lambda x: torch.permute(x, [0, 3, 1, 2]))


def nchw_nhwc(x):
    return _dfs(x, torch.Tensor, lambda x: torch.permute(x, [0, 2, 3, 1]))


class Backbone(torch.nn.Module):
    def __init__(self, backbone, permute_return=True):
        super(Backbone, self).__init__()
        self.backbone = backbone
        self.permute_return = permute_return

    def forward(self, x):
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.backbone.forward_features(x)
        if self.permute_return:
            return torch.permute(x, [0, 2, 3, 1])
        return x


class Classifer(torch.nn.Module):
    def __init__(self, backbone):
        super(Classifer, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = torch.permute(x, [0, 3, 1, 2])
        return self.backbone(x)


class AnyModule(torch.nn.Module):
    def __init__(self, module):
        super(AnyModule, self).__init__()
        self.m = module

    @staticmethod
    def perm_from_channel_last(x):
        rank = len(x.size())
        assert rank > 2 and rank < 6
        sdims = [i for i in range(1, rank - 1)]
        return x.permute([0, rank - 1, *sdims])

    @staticmethod
    def perm_to_channel_last(x):
        rank = len(x.size())
        assert rank > 2 and rank < 6
        sdims = [i for i in range(2, rank)]
        return x.permute([0, *sdims, 1])

    def forward(self, *args):
        args = _dfs(args, torch.Tensor, self.perm_from_channel_last)
        return _dfs(self.m(*args), torch.Tensor, self.perm_to_channel_last)


class Detector(torch.nn.Module):
    def __init__(self, model, post_process=False, permute_return=True):
        super(Detector, self).__init__()
        self.model = model
        self.post_process = post_process
        self.permute_return = permute_return

    def forward(self, x):
        x = torch.permute(x, [0, 3, 1, 2])
        if self.post_process:
            return self.model(x)
        if self.permute_return:
            return _flatten(
                _dfs(
                    self.model(x),
                    torch.Tensor,
                    lambda x: torch.permute(x, [0, 2, 3, 1]),
                )
            )
        ret = _flatten(self.model(x))
        return ret


def trace(module, example_input, splat=False):
    if splat:
        for v in module.parameters():
            if v.data.dtype == torch.float:
                v.data = (torch.ones_like(v) * torch.rand(1)).to(v.data.dtype)
            else:
                v.data = (torch.ones_like(v) * torch.rand(1) * 10).to(v.data.dtype)
        for v in module.buffers():
            if v.data.dtype == torch.float:
                v.data = (torch.ones_like(v) * torch.rand(1)).to(v.data.dtype)
            else:
                v.data = (torch.ones_like(v) * torch.rand(1) * 10).to(v.data.dtype)
    if isinstance(module, torch.nn.Module):
        return torch.jit.trace(module.eval(), example_input)
    else:
        return torch.jit.trace(module, example_input)
