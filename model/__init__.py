from .auto_encoder import MyModel


models_dict = {
    'my': MyModel,
}

def make_model(cfg):
    return models_dict[cfg.model_name](cfg)
