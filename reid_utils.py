"""
(ref) https://wikidocs.net/28
(ref) https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection/blob/main/run_webcam_person_detector.py
(ref) https://github.com/DoranLyong/ROS-Person-Detection-and-Tracking/blob/main/python/yolo_utils.py
"""
import sys
import os


import logging  # (ref) https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/13/logging/
from colorama import Back, Style
import torch 

from model import make_model  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/model/make_model.py

#%%
class ReID_FLAGS(object):
    def __init__(self, reid_flags):

        """ custom setup 
        """
        self.CFG_NAME = reid_flags['CUSTOM_CFG']['CFG_NAME']
        self.DATA_DIR = reid_flags['CUSTOM_CFG']['DATA_DIR']
        self.QUERY_DIR = reid_flags['CUSTOM_CFG']['QUERY_DIR']
        self.PRETRAIN_CHOICE = reid_flags['CUSTOM_CFG']['PRETRAIN_CHOICE']
        self.PRETRAIN_PATH = reid_flags['CUSTOM_CFG']['PRETRAIN_PATH']

        self.LOSS_TYPE = reid_flags['CUSTOM_CFG']['LOSS_TYPE']
        self.TEST_WEIGHT = reid_flags['CUSTOM_CFG']['TEST_WEIGHT']

        self.FLIP_FEATS = reid_flags['CUSTOM_CFG']['FLIP_FEATS']
        self.HARD_FACTOR = reid_flags['CUSTOM_CFG']['HARD_FACTOR']
        self.RERANKING = reid_flags['CUSTOM_CFG']['RERANKING']



        """ default setup 
        """
        self.PROJECT_NAME = reid_flags['DEFAULT_CFG']['PROJECT_NAME']
        self.LOG_DIR = reid_flags['DEFAULT_CFG']['LOG_DIR']
        self.OUTPUT_DIR = reid_flags['DEFAULT_CFG']['OUTPUT_DIR']
        self.DEVICE_ID = reid_flags['DEFAULT_CFG']['DEVICE_ID']

        self.LOG_PERIOD = reid_flags['DEFAULT_CFG']['LOG_PERIOD']
        self.CHECKPOINT_PERIOD = reid_flags['DEFAULT_CFG']['CHECKPOINT_PERIOD']
        self.EVAL_PERIOD = reid_flags['DEFAULT_CFG']['EVAL_PERIOD']
        self.MAX_EPOCHS = reid_flags['DEFAULT_CFG']['MAX_EPOCHS']

        # dataloader 
        self.DATALOADER_NUM_WORKERS = reid_flags['DEFAULT_CFG']['DATALOADER_NUM_WORKERS']
        self.SAMPLER = reid_flags['DEFAULT_CFG']['SAMPLER']
        self.BATCH_SIZE = reid_flags['DEFAULT_CFG']['BATCH_SIZE']
        self.NUM_IMG_PER_ID = reid_flags['DEFAULT_CFG']['NUM_IMG_PER_ID']

        # model 
        self.INPUT_SIZE = reid_flags['DEFAULT_CFG']['INPUT_SIZE']
        self.MODEL_NAME = reid_flags['DEFAULT_CFG']['MODEL_NAME']
        self.LAST_STRIDE = reid_flags['DEFAULT_CFG']['LAST_STRIDE']

        # loss 
        self.LOSS_LABELSMOOTH = reid_flags['DEFAULT_CFG']['LOSS_LABELSMOOTH']
        self.COS_LAYER = reid_flags['DEFAULT_CFG']['COS_LAYER']

        # solver 
        self.OPTIMIZER = reid_flags['DEFAULT_CFG']['OPTIMIZER']
        self.BASE_LR = reid_flags['DEFAULT_CFG']['BASE_LR']

        self.CE_LOSS_WEIGHT = reid_flags['DEFAULT_CFG']['CE_LOSS_WEIGHT']
        self.TRIPLET_LOSS_WEIGHT = reid_flags['DEFAULT_CFG']['TRIPLET_LOSS_WEIGHT']
        self.CENTER_LOSS_WEIGHT = reid_flags['DEFAULT_CFG']['CENTER_LOSS_WEIGHT']

        self.WEIGHT_DECAY = reid_flags['DEFAULT_CFG']['WEIGHT_DECAY']
        self.BIAS_LR_FACTOR = reid_flags['DEFAULT_CFG']['BIAS_LR_FACTOR']
        self.WEIGHT_DECAY_BIAS = reid_flags['DEFAULT_CFG']['WEIGHT_DECAY_BIAS']
        self.MOMENTUM = reid_flags['DEFAULT_CFG']['MOMENTUM']
        self.CENTER_LR = reid_flags['DEFAULT_CFG']['CENTER_LR']
        self.MARGIN = reid_flags['DEFAULT_CFG']['MARGIN']

        self.STEPS = reid_flags['DEFAULT_CFG']['STEPS']
        self.GAMMA = reid_flags['DEFAULT_CFG']['GAMMA']
        self.WARMUP_FACTOR = reid_flags['DEFAULT_CFG']['WARMUP_FACTOR']
        self.WARMUP_EPOCHS = reid_flags['DEFAULT_CFG']['WARMUP_EPOCHS']
        self.WARMUP_METHOD = reid_flags['DEFAULT_CFG']['WARMUP_METHOD']

        # test 
        self.TEST_IMS_PER_BATCH = reid_flags['DEFAULT_CFG']['TEST_IMS_PER_BATCH']
        self.FEAT_NORM = reid_flags['DEFAULT_CFG']['FEAT_NORM']

        self.DIST_MAT = reid_flags['DEFAULT_CFG']['DIST_MAT']
        self.PIDS = reid_flags['DEFAULT_CFG']['PIDS']
        self.CAMIDS = reid_flags['DEFAULT_CFG']['CAMIDS']
        self.IMG_PATH = reid_flags['DEFAULT_CFG']['IMG_PATH']
        self.Q_FEATS = reid_flags['DEFAULT_CFG']['Q_FEATS']  # query feats 
        self.G_FEATS = reid_flags['DEFAULT_CFG']['G_FEATS']  # gallery feats
        self.TEST_METHOD = reid_flags['DEFAULT_CFG']['TEST_METHOD']  





#%% 
class ReID_INFERENCE(object):
    def __init__(self, cfg:ReID_FLAGS, device):

        """ (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/test.py
        """
        self.log_dir = cfg.LOG_DIR
        self.logger = self.setup_logger(f'{cfg.PROJECT_NAME}.test', self.log_dir)  # print log message 
        


        """ setup 'person-reid-tiny-baseline' model 
            - example (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/tools/get_vis_result.py
            - to check if the model is train or eval mode (ref) https://discuss.pytorch.org/t/check-if-model-is-eval-or-train/9395/2
            - to check if the model is on the cuda device (ref) https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180  
        """ 
        self.model = make_model(cfg, 255)  # you can classify 255 persons 
        self.model.load_param(cfg.TEST_WEIGHT)

        self.model = self.model.to(device)
        self.model.eval()




    def model_inference(self):
        """ (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/processor/processor.py
            (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/tools/get_vis_result.py
        """
        pass 




    def setup_logger(self, name, save_dir):
        """ - (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/logger.py
            - How to use 'logging' moduel 
                ;(ref) https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/13/logging/
        """
        logger = logging.getLogger(name)  # init logger ; (ref) https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/13/logging/
        logger.setLevel(logging.DEBUG) # assign level to DEBUG message ; (ref) https://hamait.tistory.com/880

        ch = logging.StreamHandler(stream=sys.stdout)  # (ref) https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/13/logging/
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger


    
    def __call__(self, input_frame):
        """ (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/tools/get_vis_result.py
            (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/test.py
            (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/processor/processor.py
        """
        
        with torch.no_grad():
            feature = self.model(input_frame)

        

        return feature

        
        





        
        