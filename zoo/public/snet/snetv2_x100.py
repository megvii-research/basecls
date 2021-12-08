from basecls.configs.snet_cfg import SNetConfig

_cfg = dict(
    model=dict(
        name="snetv2_x100",
    ),
)


class Cfg(SNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
