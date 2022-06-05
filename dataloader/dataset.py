from torch.utils.data import Dataset
from .datainfo import Datainfo_dfaust
from config import *

def build_dataset(datatype, splits):
    if (datatype=='dfaust'):
        datainfo = Datainfo_dfaust(splits)
    else:
        print('no dataset named ' + datatype)
        exit(0)
    return MyDataset(datainfo)

class MyDataset(Dataset):
    def __init__(self, datainfo):
        self.datainfo = datainfo
    
    def __len__(self):
        return len(self.datainfo)
    
    def __getitem__(self, index):
        image = self.datainfo[index].get_image()
        surface_samples = self.datainfo[index].get_surface_samples()
        volume_samples = self.datainfo[index].get_volume_samples()
        return [image, surface_samples, volume_samples]

    def get(self, index):
        image = self.datainfo[index].get_image().unsqueeze(0)
        surface_samples = self.datainfo[index].get_surface_samples().unsqueeze(0)
        volume_samples = self.datainfo[index].get_volume_samples().unsqueeze(0)
        return [image, surface_samples, volume_samples], self.datainfo[index].path_to_mesh_file()



