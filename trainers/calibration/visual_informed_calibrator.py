import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class VisualInformedCalibration():

    def __init__(self):
        pass 

    def fit(self, val_image_features, val_text_features):

        """
        val_image_features: 
        val_text_features: 

        """
        
        image_global_center = val_image_features.mean(axis=0)
        text_global_center = val_text_features.mean(axis=0)
        self.val_image_global_center = image_global_center
        self.val_text_global_center = text_global_center


    def predict(self, logits, image_features_test, text_features_test, if_test):
        # 将 logits 和特征数据移到 CUDA
        logits = torch.from_numpy(logits).float().cuda()
        image_features_test = torch.tensor(image_features_test, dtype=torch.float32).to('cuda')
        text_features_test = torch.tensor(text_features_test, dtype=torch.float32).to('cuda')

        # 检查并将 val_image_global_center 和 val_text_global_center 转换为 PyTorch 张量
        if isinstance(self.val_image_global_center, np.ndarray):
            self.val_image_global_center = torch.tensor(self.val_image_global_center, dtype=torch.float32).to('cuda')
        if isinstance(self.val_text_global_center, np.ndarray):
            self.val_text_global_center = torch.tensor(self.val_text_global_center, dtype=torch.float32).to('cuda')

        # 计算 image 和 text 的差异
        image_differences = image_features_test - self.val_image_global_center
        text_differences = text_features_test - self.val_text_global_center

        # 计算范数
        image_distances = torch.norm(image_differences, dim=1)
        text_distances = torch.norm(text_differences, dim=1)

        # 计算分数
        image_score = torch.exp(-image_distances)
        text_score = torch.exp(-text_distances)
        # scaling_score = text_score / image_score

        # 获取预测值
        pred = logits.max(1)[1]
        scaling_score = text_score[pred] / image_score
        # print(f"scaling_score: {scaling_score}")

        # 根据 val_labels 调整 logits
        # print(f"原来的logits: {logits}")
        print("这里这里这里这里这里这里这里这里这里")
        # print(f"pred: {pred}")
        # print(f"val_labels: {val_labels}")
        original_logits = logits.clone()
        # print(f"logits.shape: {logits.shape}")
        # print(f"scaling_score.shape:{scaling_score.shape}")
        if if_test:
            # 设置打印选项以显示完整张量
            # torch.set_printoptions(profile="full")
            print(f"new: scaling_score: {scaling_score}")
            # 恢复打印设置
            # torch.set_printoptions(profile="default")
            # 检查 scaling_score 是否小于 1
            mask = scaling_score < 1
            # 使用 mask 对 logits 进行选择性缩放
            logits = torch.where(mask.unsqueeze(1), logits * scaling_score.unsqueeze(1), logits)
        # print(f"现在的logits: {logits}")
        # print(f"logits是否有改变? : {not torch.equal(original_logits, logits)}")

 
        # 将 logits 移回 CPU，并返回 NumPy 数组
        return logits.cpu().numpy()





#cpu version
    
# def difficulity_aware_calibrator(logits, class_confidences):

#     pred = logits.max(1)[1]

#     for i in range(logits.shape[0]): 
#         label = pred[i].item() 
#         confidence = class_confidences[label]
#         logits[i] *= confidence


#     return logits




