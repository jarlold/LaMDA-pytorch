import lamda_pytorch.build_streamable_dataloader
import torch

a, b = lamda_pytorch.build_streamable_dataloader.build_dataloaders()


def _get_batch_size(data):
    if isinstance(data, torch.Tensor):
        return data.size(0)
    elif isinstance(data, (list, tuple)):
        print("Ditionary!")
        if isinstance(data[0], dict):
            return data[0][list(data[0].keys())[0]].size(0)
        return data[0].size(0)
    elif isinstance(data, dict):
        print("Ditionary!")
        return data[list(data.keys())[0]].size(0)

def load_batch(data_iter):
    if data_iter is None:
        raise RuntimeError('Dataloader is not defined.')
    batch_data = next(data_iter)
    #self.batch_size = self._get_batch_size(batch_data)
    return batch_data


c = iter(a)
b = load_batch(c)
print(b['input_ids'].shape)
exit()
print(b.__class__)
print(_get_batch_size(b))



