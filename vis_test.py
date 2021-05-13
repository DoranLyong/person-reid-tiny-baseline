#%% 
import sys 
import os 
import os.path as osp 

from PIL import Image
import cv2
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



QUERY_DIR = '/home/kist-ubuntu/workspace_reID/data/Market-1501/query/' 



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



def visualizer(query_img, img_path, test_img, camid, top_k = 10, img_size=[256,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))

    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name

    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)

    cv2.imshow("results", figure)


#    pil_image=Image.fromarray(figure)  # (ref) https://www.delftstack.com/ko/howto/matplotlib/convert-a-numpy-array-to-pil-image-python/
#    pil_image.show()
    



#%%
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


    """ set dataloader 
    """
    transform = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    



    """ Test loop 
    """
    query_list = sorted(os.listdir(QUERY_DIR))
    for test_img in query_list:



        # load gallery items 
        gallery_feats = torch.load('./log/gfeats.pth').to(DEVICE) # gallery features 

        gallery_img_path = np.load('./log/imgpath.npy')

        # smaple query 
        query_img = Image.open(QUERY_DIR + test_img) 
#        query_img.show()



        input = torch.unsqueeze(transform(query_img), 0)  # [3, H, W] -> [1, 3, H, W] for torch tensor 
        input = input.to(DEVICE)


        query_feat = model(input)  # get feature in (1, 2048) tensor shape 
        norm_query = torch.nn.functional.normalize(query_feat, dim=1, p=2) # gfeats.pth 에 저장된 features도 normalized 된 상태. 
        print(f"check if normalized : {(norm_query**2).sum()} ")    # (ref) https://discuss.pytorch.org/t/question-about-functional-normalize-and-torch-norm/27755


        """ feature metrics 
        """
#        dist_mat = re_ranking(norm_query , gallery_feats , k1=30, k2=10, lambda_value=0.2) # good & slow 
        dist_mat = euclidean_distance(norm_query, gallery_feats) # not bad & fast 
#        dist_mat = cosine_similarity(norm_query, gallery_feats)  # bad & fast 

        """ Sorting 
        """ 
        indices = np.argsort(dist_mat, axis=1)  # get index order in the best order (short distnace first)
        
        print(f"Finding ID of {test_img}")
        print(f"Rank-10 gallery paths: {gallery_img_path[indices[0, :10]]}")  # Rank-10 results by slicing 10 items


        """ Visualize 
        """
        visualizer(query_img, gallery_img_path , test_img, camid='mixed', top_k = 10, img_size=[256,128])
        cv2.waitKey(32)

    cv2.destroyAllWindows()
    

# %%
