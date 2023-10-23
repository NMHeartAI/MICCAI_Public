from collections import defaultdict
# import deepcopy
from copy import deepcopy

from utils.util import *


def load_annotations(coco_path, num_imgs=1000):
    NUM_IMGS = num_imgs

    with open(coco_path, encoding="utf-8") as file:
        gt = json.load(file)

    im_anns_gt = defaultdict(list)
    for ann in gt["annotations"]:
        im_anns_gt[ann["image_id"]].append(ann)

    gt_mask = np.zeros((NUM_IMGS, 26, 512, 512), np.int32)
    for id, im in im_anns_gt.items():
        id = int(id) - 1
        for ann in im:
            points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], np.int32).T
            points = points.reshape((-1, 1, 2))
            tmp = np.zeros((512, 512), np.int32)
            cv2.fillPoly(tmp, [points], (1))
            gt_mask[id, ann["category_id"]-1] += tmp
            gt_mask[id, ann["category_id"]-1, gt_mask[id, ann["category_id"]-1] > 0] = 1

    new_gt_mask = deepcopy(gt_mask)
    for i in range(len(gt_mask)):
        file_id = int(gt['images'][i]['file_name'][:-4])
        p(f'index {i}, actual {gt["images"][i]["file_name"]}; adjusted {file_id}')
        new_gt_mask[file_id - 1] = gt_mask[i]

    return new_gt_mask

