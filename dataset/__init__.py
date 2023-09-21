from .cifar10 import MyDataset


datasets_dict = {
    'cifar10': MyDataset,
}

def make_dataset(cfg, splits=['train', 'val', 'test']):
    dataset = datasets_dict[cfg.dataset]
    splited_sets = []
    for split in splits:
        assert split in ['train', 'val', 'test']
        splited_sets.append(dataset(cfg, split))
    return splited_sets