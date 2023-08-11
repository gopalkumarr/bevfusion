from mmcv.utils import Registry
from torchpack.utils.config import configs
from mmcv import Config, DictAction
# from mmdet3d.utils import recursive_eval
import copy

__all__ = ["recursive_eval"]


import copy

__all__ = ["recursive_eval"]


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj



FUSIONMODELS = Registry("fusion_models")
# VTRANSFORMS = Registry("vtransforms")
# FUSERS = Registry("fusers")

def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
configs.load("/media/ava/DATA1/stella/default.yaml", recursive=True)
cfg = Config(recursive_eval(configs), filename="/media/ava/DATA1/stella/default.yaml")
# print(cfg)
# print(cfg.model, cfg.get("test_cfg"))
# cfg.model.train_cfg = None
# exit()
print(cfg.model)
print(type(cfg.model))

cfg = Config.fromfile("/media/ava/DATA1/stella/default.yaml",)
cfg.model.train_cfg = None
# model = build_model(cfg.model)


cfg.model.update({"type":'mmdet.ResNet'})
models = Registry('models')
mmdet_models = Registry('models', parent=models)
@mmdet_models.register_module()
class ResNet:
    pass
resnet = models.build(dict(type='mmdet.ResNet'))


# model = build_model(cfg.model)