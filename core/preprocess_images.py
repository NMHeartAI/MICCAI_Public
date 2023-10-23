from collections import defaultdict
import json, cv2
import numpy as np

from arcade_challenge_utils.logic_engine import *
from arcade_challenge_utils.utils import *
from arcade_challenge_utils.evaluate_segmentation import *
from utils.homomorphic_filter import adjust_scale
from utils.preprocess import preprocess

from utils.util import *

import os, gc, torch
from pathlib import Path
from ultralytics import YOLO

import skimage

gc.collect()
torch.cuda.empty_cache()

seg_dict = {
    i: v
    for i, v in enumerate(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "9a",
            "10",
            "10a",
            "11",
            "12",
            "12a",
            "13",
            "14",
            "14a",
            "15",
            "16",
            "16a",
            "16b",
            "16c",
            "12b",
            "14b",
            "stenosis",
        ]
    )
}

EXPERIMENTS_WITH_MODELS = [
    "_ms",
    "_ddfb",
    "_vessel_map_600epoch",
    "",
    "_bw",
    "_bwn",
    "_bwn_ms",
    "_bwn_ms_hist",
    "_bwn_ms_hist_gauss",
    "_bwn_ms_hist_gauss_fgf",
    "_bwn_ms_hist_gauss_fgf_avg"
]

single_or_multi = "multi"
epochs = 450
MODELS = {}

# ####Full Train
for experiment in EXPERIMENTS_WITH_MODELS:
    if experiment == "_stack_original_hist_fgf":
        model_path = f"/opt/app/weights/full_segmentation_dataset{experiment}_{single_or_multi}_gamma_{epochs}epochs_0.5dropout2/best.pt"
    else:
        model_path = f"/opt/app/weights/full_segmentation_dataset{experiment}_{single_or_multi}_gamma_{epochs}epochs_0.5dropout/best.pt"

    print("loading model: ", model_path)
    model = YOLO(model_path)  # load a custom model

    MODELS[experiment] = model

def pre_process_images(image_paths):

    num_images = len(image_paths)

    experiments = ['', '_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                   '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb']

    models = {k: v for k, v in MODELS.items() if k in experiments}

    try:
        os.makedirs('opt/app/saved_images/processed')
    except:
        #already exists
        pass

    #Load models
    output_matrix = np.zeros((num_images, 27, 512, 512), np.int32)

    #run models on images to get masks
    for i, full_image_path in enumerate(image_paths):
        #/*/*/image_path.png image_path is the last part of the full path
        image_path = full_image_path.split('/')[-1]
        image_id = int(image_path.split('.')[0])

        # Load original image
        original_image = np.array(Image.open(full_image_path)).copy()

        # Preprocess Image
        #homomorphic_img, nml_homomorphic_img, multi_scale_img, img_adapteq, smoothed_img, fgf_image, filtered_fgf, multi_scale_img_only, enhanced_vessels
        preprocessed_imgs = {k: adjust_scale(v) for k, v in zip(['_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                                                   '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb'],
                                                  preprocess(original_image)) if k in experiments}


        preprocessed_imgs[''] = original_image

        experiments_matrix = {}

        for experiment, model in models.items():
            temp_out = f'{image_id}.png' if experiment == '' else f'{image_id}{experiment}.png'

            image_path = f'/opt/app/saved_images/processed/{temp_out}'
            #write temp file
            cv2.imwrite(image_path, preprocessed_imgs[experiment])

            results = model.predict(image_path)[0]  # predict on an image

            experiments_matrix[experiment] = result_matrix_single(results)


        experiments_matrix['final_mask'] = np.zeros((27, 512, 512), np.uint8)

        for exp, result_mask in experiments_matrix.items():
            if exp == '_merge':
                pass
            else:
                experiments_matrix['final_mask'] = (experiments_matrix['final_mask'] == 1) | (result_mask == 1)

        # create ensemble mask
        ensemble_mask = np.zeros((len(experiments_matrix.keys()), 27, 512, 512))

        # iterate through masks and identify single class from each
        for ii, exp in enumerate(experiments_matrix.keys()):
            if exp == 'final_mask':
                pass
            else:
                new_mask = experiments_matrix[exp].argmax(axis=0)
                output_mask = np.zeros((27, 512, 512))
                for k in range(27):
                    output_mask[k] = (new_mask == k).astype(np.uint8)
                result_mask = output_mask
                ensemble_mask[ii] = result_mask
        ensemble_mask = ensemble_mask.astype(np.uint8)
        final_mask = experiments_matrix[exp] > 0

        # decide on agreement for final mask segments
        output_ = np.zeros((27, 512, 512))
        pt = final_mask.argmax(axis=0)
        for ii in range(27):
            output_[ii][(pt > 0) & (pt == ii)] = 1
        final_mask = output_

        # 27, 512, 512
        best_model_mask = ensemble_mask.argmax(axis=0)

        best_mask = np.zeros((27, 512, 512))
        for ii in range(len(experiments)):
            best_mask[(best_model_mask == ii) & (best_mask == 0)] = final_mask[
                (best_model_mask == ii) & (best_mask == 0)]
        output_ensemble = best_mask

        p(output_ensemble.max(), output_ensemble.sum(axis=0).max())

        new_output_ensemble = np.zeros(output_ensemble.shape)

        for ii in range(27):
            img = output_ensemble[ii].copy()
            new_output_ensemble[ii] = skimage.morphology.binary_dilation(img).astype(np.uint8)


        print('saving mask to /opt/app/saved_images/processed')
        output_dir = Path('/opt/app/saved_images/processed')

        # create mask
        mask = (new_output_ensemble > 0).max(axis=0)
        mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(disk_size))

        #save mask to file
        try:
            os.makedirs(f'/opt/app/saved_images/processed')
        except:
            pass

        cv2.imwrite(str(output_dir / f'{image_id}_mask.png'), mask.astype(np.uint8) * 255)


        cv2.imwrite(str(output_dir / f'{image_id}_vessel_map_600epoch.png'), preprocessed_imgs['_bwn_ms_hist_gauss_fgf'] * mask)




