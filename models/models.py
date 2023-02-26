models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make_model(args):
    model = models[args.model](args)
    return model
