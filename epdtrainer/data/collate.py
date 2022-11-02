from torch.utils.data import default_collate


class collate_list2dict:
    def __init__(self, names=['data', 'label']):
        self.names = names
        self.names_num = len(self.names)
        return

    def __call__(self, data_batch):
        out_dict = {}
        for name, data in zip(self.names, default_collate(data_batch)):
            out_dict[name] = data
        return out_dict
