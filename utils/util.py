import pickle

import torch


def print_args(args, args_file=None):
    if args_file is None:
        for k, v in sorted(vars(args).items()):
            print('{} {}'.format(k, v))
    else:
        with open(args_file, 'w') as f:
            for k, v in sorted(vars(args).items()):
                f.write('{} {}\n'.format(k, v))


def print_network(model, name, out_file=None):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    if out_file is None:
        print(name)
        print(model)
        print('The number of parameters: {}'.format(num_params))
    else:
        with open(out_file, 'w') as f:
            f.write('{}\n'.format(name))
            f.write('{}\n'.format(model))
            f.write('The number of parameters: {}\n'.format(num_params))


def save_params(params, param_file):
    with open(param_file, 'wb') as f:
        pickle.dump(params, f)


def load_params(param_file):
    with open(param_file, 'rb') as f:
        return pickle.load(f)


class InfDataLoader():
    def __init__(self, dataset, **kwargs):
        self.dataloader = torch.utils.data.DataLoader(dataset, **kwargs)

        def inf_dataloader():
            while True:
                for data in self.dataloader:
                    image, label = data
                    yield image, label

        self.inf_dataloader = inf_dataloader()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.inf_dataloader)

    def __del__(self):
        del self.dataloader
