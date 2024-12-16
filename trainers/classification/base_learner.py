import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.special import softmax


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from copy import deepcopy
from clip import clip

from tools.zsclip_encoder import build_zsclip, build_clip_templates
from tools.plot import plot_reliability_diagram
from trainers.calibration.proximity import mkdir_if_missing, get_knn_dists, get_val_image_knn_dists
from trainers.calibration.vl_calibrator import VLCalibration

import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.spatial.distance import cosine, cdist

# from trainers.calibration.base_model.coop import CustomCLIP as CoOpModel
# from trainers.calibration.base_model.cocoop import CustomCLIP as CoCoOpModel
# from trainers.calibration.base_model.kgcoop import CustomCLIP as KgCoOpModel
# from trainers.calibration.base_model.maple import CustomCLIP as MaPLeModel
# from trainers.calibration.base_model.proda import CustomCLIP as ProDAModel
# from trainers.calibration.base_model.prograd import CustomCLIP as ProgradModel
# from trainers.calibration.base_model.clip_adapter import CustomCLIP as CLIPAdapterModel
# from trainers.calibration.base_model.zsclip import CustomCLIP as ZeroShotModel
# from trainers.calibration.base_model.promptsrc import CustomCLIP as PromptSRCModel

from middlemodel_loader import get_middle_model



@TRAINER_REGISTRY.register()
class VLBaseLearner(TrainerX):
    """A base trainer for vision language tuning and calibration"""


    def after_train(self):
        print("Finish training")

        print("Testing")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()    



    # 修改after_epoch保存代码逻辑
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        # 判断是否是中间训练轮次
        is_middle_epoch = (self.epoch + 1) == (self.max_epoch // 2)

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch or is_middle_epoch:
            self.save_model(self.epoch, self.output_dir)



    @torch.no_grad()
    def test(self, split=None):
        from sklearn.metrics import accuracy_score, adjusted_rand_score
        from sklearn.cluster import KMeans
        import torch.nn.functional as F
        from sklearn.metrics import silhouette_score
        from sklearn.metrics import calinski_harabasz_score
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if not self.cfg.CALIBRATION.SCALING.IF_SCALING:  # few shot, not calibration
            if self.cfg.TRAINER.NAME == 'ProDA':
                self.model.set_classifier()

        # prepare the dataset
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader 

        print(f"Evaluate on the *{split}* set")

        # calculate the output
        image_features_test = [] 
        text_features_test = []
        all_labels = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # question04: 这里的model_inference不太懂是怎么调用的，跳转了好几次感觉都说不通。
            output, image_features_test_i, text_features_test_i = self.model_inference(input)
            # 这里的evaluator是VLClassification，process就是不断拼接（append）
            self.evaluator.process(output, label, image_features_test_i, text_features_test_i)
            image_features_test.append(image_features_test_i.data.cpu())
            all_labels.append(label.cpu().numpy())  # 保存标签

        # torch.cat(image_features_test):
        # torch.cat() 函数用于沿着指定维度连接一系列张量。
        # 在这个例子中，默认情况下它会沿着第一个维度（即批量维度）连接这些张量。
        # 这意味着所有批次的图像特征张量将被合并成一个大的张量，该张量包含了所有样本的图像特征。
        image_features_test = np.array(torch.cat(image_features_test))  
        print(f"Final image_features_test.shape = {image_features_test.shape}")
        text_features_test.append(text_features_test_i.data.cpu()) # only record once
        text_features_test = torch.cat(text_features_test, dim=0).numpy() 
        print(f"Final text_features_test.shape = {text_features_test.shape}")

        logits = np.array(self.evaluator._y_score) # logits
        # preds = np.array(self.evaluator._y_pred)
        labels = np.array(self.evaluator._y_true)
        # image_features_test = np.array(self.evaluator._image_features)
        # text_features_test = np.array(self.evaluator._text_features)

        # 感觉是为了PROCAL而存储的
        # save info from val dataloader on base class using the tuned CLIP,  and use them to train the calibrator
        if_test = False
        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            print("现在是base")
            self.save_base_val_features()
        else:
            if_test = True
            print("现在是new")
            
        # get val features on base class using tuned model for further calculation
        val_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, 'shots' + \
                                   str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, 'base', 'seed' + str(self.cfg.SEED), 'base_features.pt')
        # question03: val_dict 里面有哪些键值对？(answer03)
        # feature_dict = {
        #     "val_logits": logits_val,
        #     "val_image_features": image_feautures_val,
        #     'val_text_features': text_features_val,
        #     "val_labels": labels,
        #     "val_image_knn_dists": val_image_knn_dists
        # }
        val_dict = torch.load(val_feature_dir)
        
        # tag01: calibrator构建处
        #build the calibrator
        base_calibration_mode = self.cfg.CALIBRATION.BASE_CALIBRATION_MODE # 'scaling_based' or 'bin_based'
        base_bin_calibrator_name = self.cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME
        dac_flag = self.cfg.CALIBRATION.DAC.IF_DAC 
        procal_flag = self.cfg.CALIBRATION.PROCAL.IF_PROCAL
        # testadd
        vi_flag = self.cfg.CALIBRATION.VISUAL_INFORMED.IF_VISUAL_INFORMED
        # middleadd:
        middle_flag = self.cfg.CALIBRATION.MIDDLE_MODEL.IF_MIDDLE_MODEL
        # middleadd: middle_text_feature_dict
        middle_text_feature_dict = None
        middle_text_feature_dict_dir = osp.join('./temp/middle_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, \
                        'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED), 'middle_features.pt')
        if osp.exists(middle_text_feature_dict_dir):
            middle_text_feature_dict = torch.load(middle_text_feature_dict_dir)
        else:
            self.load_middle_model_and_save_features()
            middle_text_feature_dict = torch.load(middle_text_feature_dict_dir)
                
        if middle_text_feature_dict is not None:
            print("中间模型的文本特征成功保存并加载！！:D")
                
        val_dict = val_dict
        text_feature_dict = self.get_text_features()
        calibrator = VLCalibration(self.cfg, base_calibration_mode, base_bin_calibrator_name, dac_flag, procal_flag, vi_flag, middle_flag, val_dict, text_feature_dict, middle_text_feature_dict, if_test) # build the calibrator use val dataset
        calibrator.fit() # calibrator initialization

        # get test set proximity
        base_val_image_features = val_dict['val_image_features']
        base_dists_dir =  osp.join('./temp/knndist', self.cfg.DATASET.NAME, self.cfg.TRAINER.NAME, 'shots' + str(self.cfg.DATASET.NUM_SHOTS), \
                             self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED), 'nn' + str(self.cfg.CALIBRATION.PROCAL.IMAGE_K))   # text_knndists
        K = self.cfg.CALIBRATION.PROCAL.IMAGE_K

        dist_dir = osp.join(base_dists_dir, 'knndist.npy')   # save the test image distance for quick inference next time
        if osp.exists(dist_dir):
            print('load the knn distance from:', dist_dir)
            text_knndists = np.load(dist_dir)
        else:
            text_knndists = get_knn_dists(base_val_image_features, image_features_test, K)
            mkdir_if_missing(base_dists_dir)
            np.save(dist_dir, text_knndists)

        text_knndists = np.mean(text_knndists, axis=1) # use the average distance to K nn, TODO: need to be modified
        test_img_proximity = np.exp(-text_knndists) # knndist to proximity
        

        """ 观测指标部分 """
        # 使用KMeans 进行聚类
        all_labels = np.concatenate(all_labels, axis=0)
        print('kmeans clustering...')
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(image_features_test)
        print('kmeans clustering done')

        # 计算聚类效果
        cluster_labels = kmeans.labels_
        # print(f'Cluster Labels: {cluster_labels}')

        # 使用匈牙利算法对齐标签，精确度
        accuracy = self.cluster_accuracy(all_labels, cluster_labels)
        print(f'Clustering Accuracy (Hungarian): {accuracy * 100:.2f}%')

        # 计算调整兰德指数（ARI），类内
        ari_score = adjusted_rand_score(all_labels, cluster_labels)
        print(f'Adjusted Rand Index (ARI): {ari_score:.4f}')

        # 计算轮廓系数（Silhouette Score），类内加类外
        silhouette_avg = silhouette_score(image_features_test, all_labels)
        print(f'Silhouette Score: {silhouette_avg:.4f}')

        # 计算 Calinski-Harabasz 指数，类内加类外
        ch_score = calinski_harabasz_score(image_features_test, all_labels)
        print(f'Calinski-Harabasz Index: {ch_score:.4f}')

        # 计算相似性矩阵
        torch_text_features_test = F.normalize(torch.tensor(text_features_test), dim=1)
        torch_image_features_test = F.normalize(torch.tensor(image_features_test), dim=1)
        similarity_matrix = torch.matmul(torch_image_features_test, torch_text_features_test.T)

        # InfoNCE Loss
        temperature = 0.07
        positive_similarity = similarity_matrix[torch.arange(len(all_labels)), all_labels]
        logits_loss = similarity_matrix / temperature
        info_nce_loss = -torch.mean(torch.log(torch.exp(positive_similarity / temperature) / torch.sum(torch.exp(logits_loss), dim=1)))
        print(f'InfoNCE Loss: {info_nce_loss.item():.4f}')

        # 正负样本相似度分析
        positive_similarities = similarity_matrix[torch.arange(len(all_labels)), all_labels]
        negative_similarities = similarity_matrix.clone()
        negative_similarities[torch.arange(len(all_labels)), all_labels] = float('-inf')
        negative_similarities = torch.max(negative_similarities, dim=1).values

        positive_mean = torch.mean(positive_similarities).item()
        negative_mean = torch.mean(negative_similarities).item()
        contrast = positive_mean - negative_mean
        print(f'Positive Similarity Mean: {positive_mean:.4f}')
        print(f'Negative Similarity Mean: {negative_mean:.4f}')
        print(f'Contrast: {contrast:.4f}')
        
        """ 观测指标部分结束 """


        # confidence calibration
        probs = calibrator.predict(logits, test_img_proximity, image_features_test, text_features_test)

        # evaluate, log and plot the results
        results = self.evaluator.evaluate(probs, labels, test_img_proximity)


        for k, v in results.items():
            tag = f"{split}/{k}"
            # print(tag)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @staticmethod
    # 对齐聚类标签和真实标签
    def cluster_accuracy(true_labels, cluster_labels):
        import numpy as np        
        from scipy.optimize import linear_sum_assignment
        """
        将聚类标签重新排列，使其与真实标签尽可能匹配
        """
        D = max(cluster_labels.max(), true_labels.max()) + 1
        cost_matrix = np.zeros((D, D), dtype=np.int64)

        for i in range(len(cluster_labels)):
            cost_matrix[cluster_labels[i], true_labels[i]] += 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        return cost_matrix[row_ind, col_ind].sum() / len(cluster_labels)

    def calculate_intra_class_metrics(self, features, labels):
        grouped_features = self.group_features_by_class(features, labels)
        # print("grouped_features: ", grouped_features.shape)
        metrics = {}

        for label, class_features in grouped_features.items():
            # 确保类特征矩阵的形状正确
            if class_features.shape[0] < 2:
                continue  # 若某类样本数不足，跳过该类

            # 计算各类内的不同相关性指标
            metrics[label] = {
                "intra_class_avg_distance": self.intra_class_average_distance(class_features),
                "intra_class_variance": self.intra_class_variance(class_features),
                "std_dev_euclidean": self.std_dev_euclidean(class_features),
                "mean_similarity_to_centroid": self.mean_similarity_to_centroid(class_features)
            }
        average_dis = []
        average_variance = []
        average_dev = []
        average_sim = []

        for key in metrics:
            for metric in metrics[key]:
                if metric == "intra_class_avg_distance":
                    average_dis.append(metrics[key][metric])
                if metric == "intra_class_variance":
                    average_variance.append(metrics[key][metric])
                if metric == "std_dev_euclidean":
                    average_dev.append(metrics[key][metric])
                if metric == "mean_similarity_to_centroid":
                    average_sim.append(metrics[key][metric])

                # print("{}, {}, {}".format(key, metric, metrics[key][metric]))
        
        # please print all average value
        print(f"|{np.mean(average_dis)}|{np.mean(average_variance)}|{np.mean(average_dev)}|{np.mean(average_sim)}|")

        return metrics

    def intra_class_average_distance(self, features):
        # `features` should be an array of shape (n_samples, n_features) for one class
        return pdist(features, metric='euclidean').mean()


    def intra_class_variance(self, features):
        # features is assumed to be a (n_samples, n_features) matrix
        return np.var(features, axis=0).mean()


    def std_dev_euclidean(self, features):
        distances = pdist(features, metric='euclidean')
        return distances.std()

    def mean_similarity_to_centroid(self, features):
        centroid = features.mean(axis=0)
        distances = cdist(features, [centroid], metric='euclidean')
        return distances.mean()


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def count_unique_labels(self, dataloader):
        unique_labels = set()

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input, label = self.parse_batch_test(batch)
            unique_labels.update(label.cpu().numpy().tolist())
        print(f"There are {len(unique_labels)} unique labels in the DataLoader.")


    @torch.no_grad()
    def save_base_val_features(self):

        # only save the feature when evaluating base class 
        base_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, \
                            'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED))
        if not os.path.exists(base_dir): 
            os.makedirs(base_dir)
        save_dir = osp.join(base_dir, 'base_features.pt')

        # Check if the file already exists
        if os.path.exists(save_dir):
            print(f"File {save_dir} already exists. Skipping save operation.")
            return
        

        print("Saving base features from val dataset")
        self.set_model_mode("eval")
        

        if not self.cfg.CALIBRATION.SCALING.IF_SCALING:  # few shot, not calibration
            if self.cfg.TRAINER.NAME == 'ProDA':
                self.model.set_classifier()

        data_loader = self.val_loader # use val loader of base class
        # data_loader = self.train_loader_x

        labels = []
        image_feautures_val = []
        text_features_val = []
        logits_val = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, image_features_val_i, text_features_val_i = self.model_inference(input)
            labels.append(label.data.cpu())
            logits_val.append(output.data.cpu())
            image_feautures_val.append(image_features_val_i.data.cpu())
        
        # question05: 不太清楚 text_features_val 是否包括新类？应该不包括
        text_features_val.append(text_features_val_i.data.cpu())

        logits_val = torch.cat(logits_val, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        image_feautures_val = torch.cat(image_feautures_val, dim=0).numpy()
        text_features_val = torch.cat(text_features_val, dim=0).numpy()

        predicted_classes = np.argmax(logits_val, axis=1)
        correct_predictions = np.sum(predicted_classes == labels)
        accuracy = correct_predictions / len(labels)
        # print(f"Val Accuracy: {accuracy * 100:.2f}%")

        # save the image proximity
        val_image_knn_dists = get_val_image_knn_dists(image_feautures_val, self.cfg.CALIBRATION.PROCAL.IMAGE_K)

        # Store the info in a dictionary
        # question03 -> answer03
        feature_dict = {
            "val_logits": logits_val,
            "val_image_features": image_feautures_val,
            'val_text_features': text_features_val,
            "val_labels": labels,
            "val_image_knn_dists": val_image_knn_dists

        }
        
        torch.save(feature_dict, save_dir)


    @torch.no_grad()
    def get_text_features(self,):

        # get base val feature using tuned model,
        val_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, 'shots' + \
                                   str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME,'base', 'seed' + str(self.cfg.SEED), 'base_features.pt')
        val_dict = torch.load(val_feature_dir)
        val_text_features = val_dict['val_text_features']
        val_image_knn_dists = val_dict['val_image_knn_dists']

        # get base val feature using zero shot model
        zs_base_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  'ZeroshotCLIP', \
                                       'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, 'base', 'seed1', 'base_features.pt')
        zs_base_dict = torch.load(zs_base_feature_dir)
        
        # 1. get the base text features from zero-shot model
        base_text_features_zs = zs_base_dict['val_text_features']


        # 2. get the current text features from zero-shot model
        zs_clip  =  build_zsclip(self.cfg.MODEL.BACKBONE.NAME) # get the base model
        zs_clip.cuda()
        classnames = self.dm.dataset.classnames
        temp = build_clip_templates(self.cfg.DATASET.NAME)
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()
        with torch.no_grad():
            text_features = zs_clip.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_text_features_zs = text_features.data.cpu().numpy()


        # 3. get the base text features from tuned model
        base_text_features_tuned = val_text_features
        

        # 4. get the current text features from tuned model
        data_loader_temp = deepcopy(self.test_loader)
        batch_temp = next(iter(data_loader_temp))
        input, _ = self.parse_batch_test(batch_temp)
        _, _, current_text_features = self.model_inference(input)
        current_text_features_tuned = current_text_features.data.cpu().numpy()
        

        text_feature_dict = {
            "base_text_features_zs": base_text_features_zs,
            "current_text_features_zs": current_text_features_zs,
            'base_text_features_tuned': base_text_features_tuned,
            "current_text_features_tuned": current_text_features_tuned,

        }
        print(f"Final base_text_features_zs.shape = {base_text_features_zs.shape}")
        print(f"Final current_text_features_zs.shape = {current_text_features_zs.shape}")
        print(f"Final base_text_features_tuned.shape = {base_text_features_tuned.shape}")
        print(f"Final current_text_features_tuned.shape = {current_text_features_tuned.shape}")

        return text_feature_dict        
    
    # middleadd: 加载中间模型并保存文本特征
    def load_middle_model_and_save_features(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        middle_model = get_middle_model(cfg, classnames)

        # 怎么知道训练轮数？
        # 假设已经训练好了，并且保存了中间pth和最后pth，checkpoint显示最大轮次
        middle_epoch = self.max_epoch // 2
        # self.output_dir = "output/base2new/test_new/caltech101/shots_16/CoOp/vit_b16_c16_ep200_batch32/seed1/prompt_learner/model.pth.tar-100"
        # 不对，应该从train_base中加载
        load_base_dir = self.output_dir
        load_base_dir = load_base_dir.replace('test_new', 'train_base')
        model_file = "model.pth.tar-" + str(middle_epoch)

        if self.cfg.TRAINER.NAME == 'MaPLe':
            names = ['MultiModalPromptLearner']
        elif self.cfg.TRAINER.NAME == 'CLIP_Adapter':
            names = ['adapter']
        else:
            names = ['prompt_learner']
            
        for name in names:
            middle_model_path = osp.join(load_base_dir, name, model_file)
            checkpoint = load_checkpoint(middle_model_path)
            state_dict = checkpoint["state_dict"]

        if self.cfg.TRAINER.NAME == 'MaPLe': # load the whole model
            middle_model.load_state_dict(state_dict, strict=False)
        elif self.cfg.TRAINER.NAME == 'PromptSRC':
            middle_model.load_state_dict(state_dict, strict=False)
        else:
            state_dict_ = {}   # only load the prompt_learner
            for key, value in state_dict.items():
                print(key,'keys in base model')
                new_key = f'prompt_learner.{key}' 
                state_dict_[new_key] = value

            middle_model.load_state_dict(state_dict_, strict=False)
            # 奇怪的问题：为什么参数不匹配？？
            '''
            Traceback (most recent call last):
            File "train.py", line 467, in <module>
                main(args)
            File "train.py", line 385, in main
                trainer.test()
            File "/mnt/hdd/chenyy/conda/envs/dassl/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
                return func(*args, **kwargs)
            File "/mnt/hdd/chenyy/CLIP_Calibration/trainers/classification/base_learner.py", line 189, in test
                self.load_middle_model_and_save_features()
            File "/mnt/hdd/chenyy/CLIP_Calibration/trainers/classification/base_learner.py", line 420, in load_middle_model_and_save_features
                middle_model.load_state_dict(state_dict_, strict=False)
            File "/mnt/hdd/chenyy/conda/envs/dassl/lib/python3.8/site-packages/torch/nn/modules/module.py", line 2215, in load_state_dict
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            RuntimeError: Error(s) in loading state_dict for CustomCLIP:
                    size mismatch for prompt_learner.token_prefix: copying a param with shape torch.Size([199, 1, 512]) from checkpoint, the shape in current model is torch.Size([198, 1, 512]).
                    size mismatch for prompt_learner.token_suffix: copying a param with shape torch.Size([199, 60, 512]) from checkpoint, the shape in current model is torch.Size([198, 60, 512]).
            '''
            
        middle_model.to(self.device)
        
        middle_features_base_dir = osp.join('./temp/middle_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, \
                            'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED))
        if not os.path.exists(middle_features_base_dir):
            os.makedirs(middle_features_base_dir)
        middle_save_dir = osp.join(middle_features_base_dir, 'middle_features.pt')
        
        if os.path.exists(middle_save_dir):
            print(f"File {middle_save_dir} already exists. Skipping save operation.")
            return
        
        print("Saving middle features.")
        # 不知道要不要设置eval
        middle_model.eval()
        with torch.no_grad():
            data_loader_temp0 = deepcopy(self.val_loader)
            batch_temp0 = next(iter(data_loader_temp0))
            input0, _ = self.parse_batch_test(batch_temp0)
            # if self.dtype == torch.float16:
            #         input0 = input0.half()
            _, _, base_text_features = middle_model(input0)
            base_text_features_middle = base_text_features.data.cpu().numpy()
        

            data_loader_temp = deepcopy(self.test_loader)
            batch_temp = next(iter(data_loader_temp))
            input, _ = self.parse_batch_test(batch_temp)
            # if self.dtype == torch.float16:
            #         input = input.half()
            _, _, current_text_features = middle_model(input)
            current_text_features_middle = current_text_features.data.cpu().numpy()

        middle_text_feature_dict = {
            'base_text_features_middle': base_text_features_middle,
            "current_text_features_middle": current_text_features_middle,
        }
        
        torch.save(middle_text_feature_dict, middle_save_dir)
    
