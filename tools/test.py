import os
import torch
import sys
import math
from tqdm import tqdm
import logging

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
            cfgs.model.device = "mps"
        elif torch.cuda.is_available(): # Check if PyTorch has access to CUDA (Win or Linux's GPU architecture)
            print("CUDA is available!")
            cfgs.model.device = "cuda"
        else:
            print("Only CPU is available!")
            cfgs.model.device = "cpu"
        print(f"Using device: {cfgs.model.device}")

        self.cfgs = cfgs
        self.tile_assembly = cfgs.model.tile_assembly
        self.device = cfgs.model.device
        self.model = build_model(cfgs.model)
        Path(self.cfgs.output_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f'{cfgs.output_dir}log_testing.txt', level=logging.INFO)
        
        print(f'load ckpt from {cfgs.output_dir}')
        #ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth')
        ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth', map_location=torch.device(self.device), weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()


    def find_threshold(self):
        ann_file = open(self.cfgs.dataloader.dataset.ann_file, "rb")
        ann = pickle.load(ann_file)
        preds = list()
        targets = list()
        val_images = list()

        logging.info(f'Calculatiing "soft" predictions... ')
        print('Calculatiing "soft" predictions... ')
        for val_image in ann['val']:
            logging.info(f'Infering {val_image} ...')
            print(f'Infering {val_image} ...')
            #image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}')[:,:,0] / 255.
            image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}', cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=255.0 / image_ori.max()) / 255.
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
        
        logging.info(f'Finding threshold ...')
        print('Finding threshold ... ')
        f1s = list()
        thresholds = np.stack(list(range(40,80)))/100
        for threshold in tqdm(thresholds):
            logging.info(f'Calculating f1_score for the threshold = {threshold}')
            print(f'Calculating f1_score for the threshold = {threshold}')
            preds_ = preds.copy()
            preds_[preds_ >= threshold] = 1
            preds_[preds_ < threshold] = 0
            f1s.append(f1_score(preds_.reshape(-1), targets.reshape(-1)))
        f1s = np.stack(f1s)
        threshold = thresholds[f1s.argmax()]
        logging.info(f'Finished threshold finding. Best f1 score is {f1s.max()} at threshold = {threshold}')
        print(f'Finished threshold finding. Best f1 score is {f1s.max()} at threshold = {threshold}')
        
        return threshold


    def infer(self):
        threshold = self.find_threshold()
        #threshold = 0.69
        logging.info(f'Infering with threshold = {threshold}')
        print(f'Infering with threshold = {threshold}')
        
        folder_path = self.cfgs.dataloader.dataset.data_folder + '/../images/'
        infer_list = [f for f in os.listdir(folder_path) if 'Realistic-SBR-' in f]
        infer_list = [f.replace('Realistic-SBR-', '') for f in infer_list]
        infer_list = sorted(infer_list)
        
        # Get image size and tile size
        temp_image = cv2.imread(f'{folder_path}/Realistic-SBR-{infer_list[0]}', cv2.IMREAD_UNCHANGED)
        Im_y, Im_x = temp_image.shape
        ann_file = open(self.cfgs.dataloader.dataset.ann_file, "rb")
        ann = pickle.load(ann_file)
        val_list =  ann['val']
        temp_tile = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train2/{val_list[0]}', cv2.IMREAD_UNCHANGED)
        T, T = temp_tile.shape

        # Calculate tiling coordinates
        n_x = math.ceil(Im_x / T)
        X_coord = np.zeros(n_x, dtype=int)
        if n_x == 1:
            gap_x = 0
        else:
            gap_x = math.floor((T * n_x - Im_x) / (n_x - 1))
        gap_x_plus_one__amount = T * n_x - Im_x - gap_x * (n_x - 1)
        for i in range(1, n_x):
            if i <= gap_x_plus_one__amount:
                X_coord[i] = int(X_coord[i-1] + T - (gap_x + 1))
            else:
                X_coord[i] = int(X_coord[i-1] + T - gap_x)

        n_y = math.ceil(Im_y / T)
        Y_coord = np.zeros(n_y, dtype=int)
        if n_y == 1:
            gap_y = 0
        else:
            gap_y = math.floor((T * n_y - Im_y) / (n_y - 1))
        gap_y_plus_one__amount = T * n_y - Im_y - gap_y * (n_y - 1)
        for i in range(1, n_y):
            if i <= gap_y_plus_one__amount:
                Y_coord[i] = int(Y_coord[i-1] + T - (gap_y + 1))
            else:
                Y_coord[i] = int(Y_coord[i-1] + T - gap_y)
        
        if self.tile_assembly == 'nn': # prepare nearest neighbor map
            X_Coord = np.tile(X_coord, n_y) + (T-1) / 2
            Y_Coord = np.repeat(Y_coord, n_x) + (T-1) / 2
            y_grid, x_grid = np.meshgrid(np.arange(Im_y), np.arange(Im_x), indexing='ij')
            y_grid = y_grid[..., np.newaxis]
            x_grid = x_grid[..., np.newaxis]
            distances = np.sqrt((x_grid - X_Coord) ** 2 + (y_grid - Y_Coord) ** 2)
            nearest_map = np.argmin(distances, axis=-1)

        Path(f'{self.cfgs.output_dir}/evaluation_{self.tile_assembly}').mkdir(parents=True, exist_ok=True)
        for image_name in infer_list:
            logging.info(f'Infering {image_name} ...')
            print(f'Infering {image_name} ...')
            image_ori = cv2.imread(f'{folder_path}/Realistic-SBR-{image_name}', cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=255.0 / image_ori.max()) / 255.

            # Inference by tiles
            pred_array = np.zeros((n_x * n_y, Im_y, Im_x), dtype=np.float32)
            for i in range(n_y):
                for j in range(n_x):
                    tile = image_ori[Y_coord[i]:(Y_coord[i] + T), X_coord[j]:(X_coord[j] + T)] # Crop the ROI
                    # Start the infering process
                    tile_flip_0 = cv2.flip(tile, 0)
                    tile_flip_1 = cv2.flip(tile, 1)
                    tile_flip__1 = cv2.flip(tile, -1)
                    tile_stack = np.stack([tile, tile_flip_0, tile_flip_1, tile_flip__1])
                    tile_torch = torch.tensor(tile_stack).unsqueeze(1).to(torch.float32).to(self.device)
                    with torch.no_grad():
                        pred, _, _, _ = self.model(tile_torch)
                        pred = torch.sigmoid(pred)
                        pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred
                    pred_ori = pred_ori.cpu().numpy()
                    pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy(), 0)
                    pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy(), 1)
                    pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy(), -1)
                    tile_pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1], axis=0)
                    pred_array[i * n_x + j, Y_coord[i]:(Y_coord[i] + T), X_coord[j]:(X_coord[j] + T)] = tile_pred
                    
            # Averaging the result
            non_zero_mask = pred_array != 0  # Shape (n_x * n_y, img_height, img_width)
            non_zero_count = np.sum(non_zero_mask, axis=0)  # Shape (img_height, img_width)
            non_zero_count[non_zero_count == 0] = 1  # Prevent division by zero
            if self.tile_assembly == 'mean':
                non_zero_sum = np.sum(pred_array * non_zero_mask, axis=0)  # Shape (img_height, img_width)
                pred = non_zero_sum / non_zero_count  # Shape (img_height, img_width)
            elif self.tile_assembly == 'max':
                pred = np.max(pred_array * non_zero_mask, axis=0)
            elif self.tile_assembly == 'nn': # nearest neighbor
                pred = np.zeros((Im_y, Im_x), dtype=np.float32)
                for idx in range(n_y * n_x):
                    pred[nearest_map == idx] = pred_array[idx, nearest_map == idx]
            else:
                pred = np.zeros((Im_y, Im_x), dtype=np.float32)
                raise ValueError(f"Unknown tile assembly method: {self.tile_assembly}")
            
            # Saving results
            evaluation_folder = 'evaluation_' + self.tile_assembly
            image_ori = (image_ori * 255).astype(np.uint8)
            cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_orig.tif"}', image_ori)
            
            pred_bin = pred.copy()
            pred_bin[pred_bin >= threshold] = 1
            pred_bin[pred_bin < threshold] = 0
            pred_bin = (pred_bin * 255).astype(np.uint8)
            cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_pred_bin.tif"}', pred_bin)
            
            pred = (pred * 255).astype(np.uint8)
            cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_pred.tif"}', pred)
            target = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/../images/Skeleton{image_name[1:]}')[:,:,0]
            #target = (target * 255).astype(np.uint8)
            cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_target.tif"}', target.astype(np.uint8))
            
            ## Analyze the difference between original and binarized prediction
            #diff_pred = np.zeros_like(pred, dtype=np.uint8)
            #diff_pred[pred_bin != pred] = 255
            #cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_diff_pred.tif"}', diff_pred)
            
            # Highlight the difference between prediction and target
            diff = np.stack([pred_bin] * 3, axis=-1).astype(np.uint8)       
            diff[(pred_bin == 255) & (target == 0)] = [0, 255, 0]  # Green: appears in prediction, missing in target
            diff[(pred_bin == 0) & (target == 255)] = [255, 0, 0]  # Red: issing in prediction, appears in target
            cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + "_diff.tif"}', cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
            
            # # Create a concatenated image for visual inspection
            # cat1 = np.concatenate([image_ori, pred, diff_pred], axis=1)
            # cat1 = np.stack([cat1] * 3, axis=-1).astype(np.uint8)
            # cat2 = np.concatenate([diff, np.stack([pred_bin] * 3, axis=-1).astype(np.uint8), np.stack([target] * 3, axis=-1).astype(np.uint8)], axis=1)
            # cat = np.concatenate([cat1, cat2], axis=0)
            # cv2.imwrite(f'{self.cfgs.output_dir}/{evaluation_folder}/{os.path.splitext(image_name)[0] + ".tif"}', cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))

        logging.info(f'Done.')
        print('Done.')
    
    
    def infer_tta_6(self):
        threshold = self.find_threshold()
        print(f'Inferencing with threshold = {threshold}')
        
        ann_file = open(self.cfgs.dataloader.dataset.ann_file, "rb")
        ann = pickle.load(ann_file)
        #test_list = ann['val']
        test_list = []
        
        print('Testing on the whole set ... \n')
        Path(f'{self.cfgs.output_dir}/submission').mkdir(parents=True, exist_ok=True)
        for test_image_name in test_list:
            print(f"Processing {test_image_name}...")
            
            #image_ori = cv2.imread(image_path)
            #image_ori = (image_ori[:,:,0]/255.)
            image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{test_image_name}', cv2.IMREAD_UNCHANGED)
            image_ori = cv2.convertScaleAbs(image_ori, alpha=255.0 / image_ori.max()) / 255.

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

            pred[pred >= threshold] = 1
            pred[pred < threshold] = 0
            pred = pred.astype(np.uint8) * 255
            cv2.imwrite(f'{self.cfgs.output_dir}/submission/{os.path.splitext(test_image_name)[0] + ".tif"}', pred)

        print('done')
