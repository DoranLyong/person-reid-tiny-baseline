
#%% 
import sys 
import os 
import os.path as osp 

import numpy as np 
import yaml 
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import torch 
import torchvision.transforms as T

from datasets import make_dataloader
from utils.reranking import re_ranking  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/reranking.py
from utils.metrics import eval_func
from reid_utils import (ReID_FLAGS, 
                        ReID_INFERENCE,
                        )


""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = os.getcwd()
os.chdir(cwd) #changing working directory 

print(f"Python version: {Back.GREEN}{python_ver}{Style.RESET_ALL}")
print(f"The path of the running script: {Back.MAGENTA}{script_path}{Style.RESET_ALL}")
print(f"CWD is changed to: {Back.RED}{cwd}{Style.RESET_ALL}")



#%% Load configurations in YAML
try: 
    with open('config/reid_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print("YAML is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML...")


reid_flags = ReID_FLAGS(cfg)
#print(vars(reid_flags))  # check the class members(ref) https://www.programiz.com/python-programming/methods/built-in/vars






#%% 
def vis_tensorImg(img:torch.Tensor):
    """ - tensor img to PIL (ref) https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/2
        - show normalized tensor image (ref) https://discuss.pytorch.org/t/conversion-from-a-tensor-to-a-pil-image-not-working-well-what-is-going-wrong/26121/2
    """
    img = img.to('cpu')    
    
    # Denormalize the tensor image 
    std = [0.229, 0.224, 0.225] # get 'std' from (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/make_dataloader.py
    mean = [0.485, 0.456, 0.406]
    z = img * torch.tensor(std).view(3, 1, 1)  
    z = z + torch.tensor(mean).view(3, 1, 1)

    img_pil = T.ToPILImage()(z).convert("RGB")
    img_pil.show()



#%% Metrics 
def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()



def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat





#%% 
if __name__ == '__main__':

    """ Set your device 
    """
    gpu_no = 0  # gpu_number 
    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: { DEVICE }")    


    """ init. inference object  
    """
    model = ReID_INFERENCE(reid_flags, DEVICE)


    """ data loader 
    """
    train_loader, val_loader, num_query, num_classes = make_dataloader(reid_flags)  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/make_dataloader.py

    img_path_list = []
    feats, pids, camids = [], [], []

    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):

        img = img.to(DEVICE)  # get one input data 

        
#        vis_tensorImg(img[0])  # for checking images  
        feat = model(img)  # get feature in (Batch_size, 2048) tensor shape 



        """ update list
        """ 
        img_path_list.extend(imgpath)   # (ref) https://wikidocs.net/14#extend
                                        # extend vs. append (ref) https://www.edureka.co/community/5916/difference-between-append-vs-extend-list-methods-in-python  
        pids.extend(np.asarray(pid))
        camids.extend(np.asarray(camid))
        feats.append(feat)  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/metrics.py
        


    """ Compute Q_FEATS, G_FEATS
        (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/metrics.py
    """
    feats_tensor = torch.cat(feats, dim=0) # [N, 2048]
    norm_feats = torch.nn.functional.normalize(feats_tensor, dim=1, p=2)    # along channel ; (ref) https://pytorch.org/docs/master/generated/torch.nn.functional.normalize.html
                                                                            # p=2 for L2-norm 

    # query 
    qf = norm_feats[:num_query]  # query_features; 여기까지는 query 데이터로 사용 
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    # gallery
    gf = norm_feats[num_query:] # gallery_features; 나머지는 gallery 
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])


    # Reranking by similarity ; (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/metrics.py
    """ (opt1) re-ranking algorithm 
        (opt2) Euclidean 
        (opt3) cosine_similarity 
    """
#    distmat = re_ranking(qf, gf, k1=30, k2=10, lambda_value=0.2)  # good??? & slow 
    distmat = euclidean_distance(qf, gf)  # good & fast 
#    distmat = cosine_similarity(qf, gf) # bad & slow 

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)


    """ Save metrics
        (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/processor/processor.py
    """
    print(f"Save metrics")
    np.save(osp.join('./log', 'dist_mat.npy') , distmat)
    np.save(osp.join('./log', 'pids.npy') , pids)
    np.save(osp.join('./log', 'camids.npy') , camids)
    np.save(osp.join('./log', 'imgpath.npy') , img_path_list[num_query:]) # gallery image paths

    torch.save(qf, osp.join('./log', 'qfeats.pth' ))
    torch.save(gf, osp.join('./log', 'gfeats.pth' ))
    



    

# %%
