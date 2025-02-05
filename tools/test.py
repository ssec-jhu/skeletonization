import os
import torch
import sys
from tqdm import tqdm

sys.path.append('.')
from model import build_model
from pathlib import Path
from glob import glob
import cv2
import pickle
import numpy as np
from sklearn.metrics import f1_score

class Tester:
    def __init__(self, cfgs):
        
        # Find available GPU
        if torch.backends.mps.is_available(): # Check if PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
            print("MPS is available!")
            if torch.backends.mps.is_built():
                print("MPS (Metal Performance Shader) is built in!")    
            device = "mps"
        elif torch.cuda.is_available(): # Check if PyTorch has access to CUDA (Win or Linux's GPU architecture)
            print("CUDA is available!")
            device = "cuda"
        else:
            print("Only CPU is available!")
            device = "cpu"
        print(f"Using device: {device}")

        self.cfgs = cfgs
        self.device = device # cfgs.model.device
        cfgs.model.device = device
        self.model = build_model(cfgs.model)
        Path(f'{self.cfgs.output_dir}/submission').mkdir(parents=True, exist_ok=True)
        Path(f'{self.cfgs.output_dir}/evaluation').mkdir(parents=True, exist_ok=True)

        print(f'load ckpt from {cfgs.output_dir}')
        #ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth')
        #ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth', map_location=torch.device('cpu'), weights_only=False)
        ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth', map_location=torch.device(self.device), weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()


    def find_threshold(self):
        ann_file = open(self.cfgs.dataloader.dataset.ann_file, "rb")
        ann = pickle.load(ann_file)
        preds = list()
        targets = list()
        val_images = list()
        
        print('testing on validation set ... ')
        for val_image in ann['val']:
            #image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}')[:,:,0] / 255.
            image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}', cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=255.0 / image_ori.max()) / 255.
            #cv2.imshow('Image after', image_ori)
            #cv2.waitKey(0)  # Waits indefinitely; use cv2.waitKey(1000) to wait 1 second
            #cv2.destroyAllWindows()
            target = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train2/{val_image}')[:,:,0] / 255.

            image_flip_0 = cv2.flip(image_ori, 0)
            image_flip_1 = cv2.flip(image_ori, 1)
            image_flip__1 = cv2.flip(image_ori, -1)
            image = np.stack([image_ori, image_flip_0, image_flip_1, image_flip__1])

            #image = torch.tensor(image).unsqueeze(1).to(self.device)
            image = torch.tensor(image).unsqueeze(1).to(torch.float32).to(self.device)
            with torch.no_grad():
                pred, _, _, _ = self.model(image)
                pred = torch.sigmoid(pred)
                pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred

            pred_ori = pred_ori.cpu().numpy()
            pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy(), 0)
            pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy(), 1)
            pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy(), -1)
            pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1], axis=0)

            preds.append(pred)
            targets.append(target)
            val_images.append(val_image)
        
        preds = np.stack(preds)
        targets = np.stack(targets)

        print('finding threshold ... ')
        f1s = list()
        thresholds = np.stack(list(range(40,80)))/100
        for threshold in tqdm(thresholds):
            preds_ = preds.copy()
            preds_[preds_ >= threshold] = 1
            preds_[preds_ < threshold] = 0
            f1s.append(f1_score(preds_.reshape(-1), targets.reshape(-1)))
        f1s = np.stack(f1s)
        threshold = thresholds[f1s.argmax()]
        print(f'best f1 score is {f1s.max()} at threshold = {threshold}')

        # write valid results
        diff_pred_arr = list()
        diff_target_arr = list()
        for val_image, pred, target in zip(val_images, preds, targets):
            image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}', cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=255.0 / image_ori.max()).astype(np.uint8)
            
            pred[pred >= threshold] = 1
            pred[pred < threshold] = 0
            pred = pred.astype(np.uint8) * 255
            
            target = (target * 255).astype(np.uint8)
            
            cat = np.concatenate([image_ori, pred, target], axis=1)
            cv2.imwrite(f'{self.cfgs.output_dir}/evaluation/{os.path.splitext(val_image)[0] + "_0.tif"}', cat) #, cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))
            
            # Set the differences
            diff = np.zeros_like(target, dtype=np.uint8)
            diff[((pred == 255) & (target == 255)) | ((pred == 0) & (target == 0))] = 128  # Both have the same value
            diff[(pred == 255) & (target == 0)] = 255  # Appears in prediction, missing in target
            diff[(pred == 0) & (target == 255)] = 0      # Missing in prediction, appears in target
            diff = np.stack([diff] * 3, axis=-1)
            
            diff_pred = np.stack([pred] * 3, axis=-1).astype(np.uint8)       
            diff_pred[(pred == 255) & (target == 0)] = [255, 0, 0]  # Appears in prediction, missing in target
            diff_pred_arr.append(np.sum(np.all(diff_pred == [255, 0, 0], axis=-1)))

            diff_target = np.stack([target] * 3, axis=-1).astype(np.uint8)
            diff_target[(pred == 0) & (target == 255)] = [255, 0, 0]  # Missing in prediction, appears in target
            diff_target_arr.append(np.sum(np.all(diff_target == [255, 0, 0], axis=-1)))
            
            diff = np.concatenate([diff, diff_pred, diff_target], axis=1)
            cv2.imwrite(f'{self.cfgs.output_dir}/evaluation/{os.path.splitext(val_image)[0] + "_1.tif"}', cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
            
            cat = np.stack([cat] * 3, axis=-1)
            cat = np.concatenate([cat, diff], axis=0)
            cv2.imwrite(f'{self.cfgs.output_dir}/evaluation/{os.path.splitext(val_image)[0] + ".tif"}', cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))

        return threshold


    def infer_tta_6(self):
        threshold = self.find_threshold()
        #threshold = 74
        print(f'inferencing with threshold = {threshold}')
        
        
        #test_list = glob(f'{self.cfgs.dataloader.dataset.data_folder}/../../../CIV_Developmental_Images/24hr/Processed/Oriented/*tif')
        test_list = []
        for image_path in test_list:
            image_name = image_path.split('/')[-1]

            #image_ori = cv2.imread(image_path)
            #image_ori = (image_ori[:,:,0]/255.)
            image_ori = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=1.0 / image_ori.max())
            
            image_flip_0 = cv2.flip(image_ori, 0)
            image_flip_1 = cv2.flip(image_ori, 1)
            image_flip__1 = cv2.flip(image_ori, -1)
            image_rotate_90cc = cv2.rotate(image_ori, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_rotate_90c = cv2.rotate(image_ori, cv2.ROTATE_90_CLOCKWISE)
            image_rotate_180 = cv2.rotate(image_ori, cv2.ROTATE_180)
            image = np.stack([image_ori, image_flip_0, image_flip_1, image_flip__1, image_rotate_90cc, image_rotate_90c, image_rotate_180])

            #image = torch.tensor(image).unsqueeze(1).to(self.device)
            image = torch.tensor(image).unsqueeze(1).to(torch.float32).to(self.device)
            with torch.no_grad():
                pred, _, _, _ = self.model(image)
                pred = torch.sigmoid(pred)
                pred_ori, pred_flip_0, pred_flip_1, pred_flip__1, pred_rotate_90cc, pred_rotate_90c, pred_rotate_180 = pred

            pred_ori = pred_ori.cpu().numpy()
            pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy(), 0)
            pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy(), 1)
            pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy(), -1)
            pred_rotate_90cc = cv2.rotate(pred_rotate_90cc.cpu().numpy(), cv2.ROTATE_90_CLOCKWISE)
            pred_rotate_90c = cv2.rotate(pred_rotate_90c.cpu().numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            pred_rotate_180 = cv2.rotate(pred_rotate_180.cpu().numpy(), cv2.ROTATE_180)

            pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1, pred_rotate_90cc, pred_rotate_90c, pred_rotate_180], axis=0)
            pred = np.stack([pred, pred, pred], axis=2)

            pred[pred >= threshold] = 255
            pred[pred < threshold] = 0
            cv2.imwrite(f'{self.cfgs.output_dir}/submission/{image_name}', pred)

        print('done')
