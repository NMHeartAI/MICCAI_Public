#Syntax Project Utils
#Tom Liu

"""Utility functions for videos, plotting and computing performance metrics."""

import functools
import json
import os

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage import measure
# import precision_recall_curve
from sklearn.metrics import precision_recall_curve, auc, roc_curve

global verbose
verbose = True

def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array

def toggle_verbose(verbosity=None):
    global verbose
    verbose = not verbose if not verbosity else verbosity

def verbose_mode(func, verbose_override=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global verbose
        if verbose_override or verbose is True:
            res = func(*args, **kwargs)
        else:
            res = None
        return res

    return wrapper

@verbose_mode
def p(*args, func=print):
    return func(*args)


def dice_score(x, y):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """

    inter = x * y

    return (2 * np.sum(inter)) / (np.sum(x) + np.sum(y))

# __all__ = ["video", "segmentation", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]

def generate_resut_json(empty_json_path, result_json_path, result_mask, mode):
    
    with open(empty_json_path) as file:
        gt = json.load(file)
    empty_submit = dict()
    empty_submit["images"] = gt["images"]
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    
    gt_mask = result_mask
    
    count_anns = 1
    for img_id, img in enumerate(gt_mask, 0):
        for cls_id, cls in enumerate(img, 0):
            contours = measure.find_contours(cls)
            for contour in contours:            
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
            
                if(poly.is_empty):
                    continue
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                new_ann = dict()
                new_ann["id"] = count_anns
                new_ann["image_id"] = img_id+1
                if mode=='seg':
                  new_ann["category_id"] = cls_id+1
                else:
                  new_ann["category_id"] = 26
                new_ann["segmentation"] = [segmentation]
                new_ann["area"] = poly.area
                x, y = contour.min(axis=0)
                w, h = contour.max(axis=0) - contour.min(axis=0)
                new_ann["bbox"]  = [int(x), int(y), int(w), int(h)]
                new_ann["iscrowd"] = 0
                # new_ann["attributes"] = {"occluded": False}
                count_anns += 1
                empty_submit["annotations"].append(new_ann.copy())
   
    with open(result_json_path, "w") as file:
        json.dump(empty_submit, file)


