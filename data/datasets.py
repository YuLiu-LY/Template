
datamodules = {}

def register(name):
    def decorator(cls):
        datamodules[name] = cls
        return cls
    return decorator

def make_datamodule(args):
    datamodule = datamodules[args.dataset](args)
    return datamodule
