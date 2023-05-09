models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make_model(model_name, args):
    model = models[model_name](args)
    return model
