from .classifier import ResModel, MyModel


models_dict = {
    'resnet': ResModel,
    'my': MyModel,
}

def make_model(cfg):
    return models_dict[cfg.model_name](cfg)
