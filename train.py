import torch
from torch.utils.data import Dataset
from utils import parse_args



class dict_dataset(Dataset):
    '''
    将list装换为 torch dataset
    '''
    def __init__(self, dict_data):
        self.data = dict_data.values()

        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def main():
    args = parse_args()
    print(args.sememe_data_path)