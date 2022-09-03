import torch
import pickle
import json 
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dnp(tensor,gpu = False):
    # detach the tensor from the calculation tree
    if gpu: return tensor.cpu().detach().numpy()
    return tensor.detach().numpy()

def progress_bar(count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
# load json data
def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return data

def save_json(data,path):
    '''input the diction data and save it'''
    beta_file = json.dumps(data)
    file = open(path,'w')
    file.write(beta_file)
    return True

# load pickle data
def save_pickle(data,name):
    with open(str(name), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_name):
    pkl_file = open(file_name, 'rb')
    data = pickle.load(pkl_file)
    return data