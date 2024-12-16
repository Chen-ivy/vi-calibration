import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os.path as osp
import os


class MiddleModelCalibration():

    def __init__(self):
        pass 

    def fit(self, text_feature_dict, base_text_features_middle, current_text_features_middle):

        """
        text_feature_dict: 
        base_text_features_middle: 
        current_text_features_middle: 

        """
        base_text_features_zs = text_feature_dict["base_text_features_zs"]
        current_text_features_zs = text_feature_dict["current_text_features_zs"]
        base_text_features_tuned = text_feature_dict["base_text_features_tuned"]
        current_text_features_tuned = text_feature_dict["current_text_features_tuned"]
        
        class_confidence = []
        cur_class_num = current_text_features_middle.shape[0]
        threshold = 0.05  # 设定一个微小误差的阈值（根据DAC设定）
        
        for i in range(cur_class_num):
            # 判断是不是base类
            distances = np.linalg.norm(base_text_features_zs - current_text_features_zs[i], axis=1)
            is_base_class = np.any(distances < threshold)
            if is_base_class:
                class_confidence_i = 1.0
            else:
                sim_zs_middle = self.cosine_similarity(current_text_features_zs[i], current_text_features_middle[i])
                sim_zs_tuned = self.cosine_similarity(current_text_features_zs[i], current_text_features_tuned[i])      
                # TODO: 找到真正好的放缩参数
                # 1. 已经尝试：class_confidence_i = sim_zs_middle / sim_zs_tuned  
                #    效果很差
                # 2. 已经尝试：class_confidence_i = (1 - sim_zs_middle) / (1 - sim_zs_tuned)
                #    效果一般，但是会比DAC差
                class_confidence_i = sim_zs_middle / sim_zs_tuned  
            
            class_confidence.append(class_confidence_i)
        
        print("打印 class_confidence: ")
        print(class_confidence)

        self.class_confidence = np.array(class_confidence)
        
    def cosine_similarity(self, a, b):
        # 计算点积
        dot_product = np.dot(a, b)
        # 计算模长
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # 返回余弦相似度
        return dot_product / (norm_a * norm_b)     



    def predict(self, logits):
        # gpu version to acc inference
        logits = torch.from_numpy(logits).float().cuda()
        class_confidences = torch.from_numpy(self.class_confidence).float().cuda()

        pred = logits.max(1)[1]
        logits *= class_confidences[pred][:, None]

        return logits.cpu().numpy()





#cpu version
    
# def difficulity_aware_calibrator(logits, class_confidences):

#     pred = logits.max(1)[1]

#     for i in range(logits.shape[0]): 
#         label = pred[i].item() 
#         confidence = class_confidences[label]
#         logits[i] *= confidence


#     return logits




