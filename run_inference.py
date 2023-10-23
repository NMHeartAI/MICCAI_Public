from arcade_challenge_utils.logic_engine import *
from arcade_challenge_utils.utils import *
from arcade_challenge_utils.evaluate_segmentation import *
from core.preprocess_images import MODELS
from utils.homomorphic_filter import adjust_scale
from utils.preprocess import preprocess

import skimage
from skimage.draw import polygon

import gc, torch
from pathlib import Path
from ultralytics import YOLO

import json
from datetime import datetime

# Get the current date and time
current_datetime = datetime.today()

# Convert datetime to ISO 8601 format
iso_datetime = current_datetime.isoformat()

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

exclude_segments = {
    k + 1: v
    for k, v in {
        # 9: "9a",
        10: "10",
        11: "10a", #doesnt exist in dataset
        13: "12", #tidesing has it
        14: "12a",
        #  15: "13",
        16: "14",
        #17: "14a",#tidesing has it
        18: "15",
        20: "16a",
        21: "16b",
        22: "16c",
        23: "12b",
        24: "14b",
    }.items()
}

post_exclusion_segments = {
    k + 1: v
    for k, v in {
        9: "9a",
        # 10: "10",
        13: "12",
        14: "12a",
        # 15: '13',
        # 16: "14",
        17: "14a",
        18: "15",
        20: "16a",
        21: "16b",
        22: "16c",
        23: "12b",
        24: "14b",
    }.items()
}

single_or_multi = "multi"
num_images = 200; NUM_IMGS = num_images
# Garbage Collect
gc.collect()
torch.cuda.empty_cache()
confidence = 0.25

cat_order = np.arange(26).astype(int)

experiments = [
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
]

experiments_matrix = {}

arcade = Path("/data2/arcade_challenge/arcade_dataset_phase_1")

models = {k: v for k, v in MODELS.items() if k in experiments}

def size_filter(output_matrix, j):
    for i in range(27):
        area = output_matrix[j - 1, i].sum()

        if i > 0:
            print(f"evaluating area for segment {seg_dict[i-1]}, {i}: {area}")

        if area > 0:
            if ((seg_dict[i - 1] == "4")) and (area < 800):
                print(
                    f"excluding segment {seg_dict[i-1]} because area {area} is too small"
                )
                output_matrix[j - 1, i] = np.zeros((512, 512))
            if ((seg_dict[i - 1] == "16")) and (area < 800):
                print(
                    f"excluding segment {seg_dict[i-1]} because area {area} is too small"
                )
                output_matrix[j - 1, i] = np.zeros((512, 512))
            if area < 400:
                if (area < 200) and (seg_dict[i - 1] == "5"):
                    continue
                elif seg_dict[i - 1] == "5":
                    print(
                        f"5 principal axis {get_principal_axis(get_contour_poly(output_matrix[j - 1, i]))}"
                    )
                else:
                    print(
                        f"excluding segment {seg_dict[i-1]} because area {area} is too small"
                    )
                    output_matrix[j - 1, i] = np.zeros((512, 512))


        # Assuming you have already obtained the contour as a list of coordinates

        contours = measure.find_contours(output_matrix[j - 1][i])

        print('number of contours found:', len(contours))

        if len(contours) > 0:
            contour = max(contours, key=lambda x: len(x))

            print('length of contour:', len(contour))

            # Create an empty 512x512 NumPy array
            image_shape = (512, 512)
            filled_image = np.zeros(image_shape, dtype=np.uint8)

            # Convert the contour coordinates into integer values and create a filled polygon
            rr, cc = polygon(contour[:, 0], contour[:, 1], image_shape)

            # Fill the region inside the contour in the empty image
            filled_image[rr, cc] = 1  # You can use any value you want to fill the region

            output_matrix[j - 1, i] = filled_image

            # print([len(measure.find_contours(output_matrix[j - 1][i])) for i in range(27)])

    return output_matrix

def export_json(output_matrix, num_images=30, output_json_path='/opt/app/json_out/coronary-artery-segmentation.json', mapping_dict=dict()):
    with open('/opt/app/json_out/empty_annotations.json') as file:
        gt = json.load(file)

    gt_images = [{
                "id": i,
                "width": 512,
                "height": 512,
                "file_name": f"{i}.png",
                "license": 0,
                "date_captured": iso_datetime
                }
                for i in range(1, num_images+1)]

    print('writing:',  {f'{i}.png': mapping_dict.get(f"{i}.png", f"{i}.png") for i in range(1, num_images+1)})

    info = {'description': 'NU Team 2023',
           'version': 'v1',
           'year': 2023,
           'contributor': 'txlmd',
            'date_created': datetime.today().strftime("%Y-%m-%d")}

    licenses = [{'id': i,
                 'name': mapping_dict.get(f"{i}.png", f"{i}.png"),
                 'url': "https://www.feinberg.northwestern.edu/"} for i in range(1, num_images+1)]

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    empty_submit["info"] = info
    empty_submit["licenses"] = licenses

    gt_mask = output_matrix[0:num_images, 1:27, ...]

    count_anns = 1
    areas = []
    for img_id, img in enumerate(gt_mask, 0):
        for cls_id, cls in enumerate(img, 0):
            contours = measure.find_contours(cls)

            for contour in contours:
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)

                if poly.is_empty:
                    continue
                if poly.geom_type == "Polygon":
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                elif poly.geom_type == "MultiPolygon":
                    poly = poly.simplify(1.0, preserve_topology=False)
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()

                    if not poly.is_valid:
                        # Attempt to fix self-intersections using buffering
                        try:
                            buffered_poly = poly.buffer(0)
                            if buffered_poly.is_valid:
                                poly = buffered_poly
                        except:
                            raise Exception("Sorry, no numbers below zero")

                # filter out small segments
                if poly.area > 250:
                    new_ann = dict()

                    new_ann["id"] = count_anns
                    new_ann["image_id"] = img_id + 1
                    new_ann["category_id"] = cls_id + 1
                    new_ann["segmentation"] = [segmentation]
                    new_ann["area"] = poly.area
                    x, y = contour.min(axis=0)
                    w, h = contour.max(axis=0) - contour.min(axis=0)
                    new_ann["bbox"] = [int(x), int(y), int(w), int(h)]
                    new_ann["iscrowd"] = 0
                    # new_ann["attributes"] = {"occluded": False}
                    count_anns += 1
                    empty_submit["annotations"].append(new_ann.copy())
                    areas.append(poly.area)

    with open(output_json_path, "w") as file:
        json.dump(empty_submit, file)

    # print(output_json_path)
    return output_json_path

def erode_n(img, n=1):
    if n == 0:
        return img
    return erode_n(skimage.morphology.binary_erosion(img), n-1) #vessel_mask_map = skimage.morphology.binary_erosion(vessel_mask_map)img


def create_mask_from_model_output(mask_matrix):
    mask_matrix['final_mask'] = np.zeros((27, 512, 512), np.uint8)

    for exp, result_mask in mask_matrix.items():
        if exp == '_merge':
            pass
        else:
            mask_matrix['final_mask'] = (mask_matrix['final_mask'] == 1) | (result_mask == 1)

    ensemble_mask = np.zeros((len(mask_matrix.keys()), 27, 512, 512))

    for ii, exp in enumerate(mask_matrix.keys()):
        if exp == 'final_mask':
            pass
        else:
            new_mask = mask_matrix[exp].argmax(axis=0)
            output_mask = np.zeros((27, 512, 512))
            for k in range(27):
                output_mask[k] = (new_mask == k).astype(np.uint8)
            result_mask = output_mask
            ensemble_mask[ii] = result_mask
    ensemble_mask = ensemble_mask.astype(np.uint8)
    final_mask = mask_matrix[exp] > 0

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

    mask = (new_output_ensemble > 0).max(axis=0)

    return mask

def logical_construction_validation_set(path_to_images, mapping_dict=dict()):

    inverse_dict = dict() #{v:k for k, v in mapping_dict.items()} if len(mapping_dict) > 0 else dict()

    output_json_path = f"/opt/app/json_out/coronary-artery-segmentation.json"

    num_images = len(path_to_images)
    # Load models
    output_matrix = np.zeros((num_images, 27, 512, 512), np.float32)

    # run models on images to get masks full imagepath is n.png
    for full_image_path in path_to_images:

        image_path = full_image_path.split('/')[-1]
        #n.png --> slice_x.png 0png --> 1
        img_id = int(image_path.split('.')[0]); j = img_id

        print(f'processing image: {image_path} | {inverse_dict.get(image_path, image_path)}')

        # Load original image
        original_image = np.array(Image.open(full_image_path)).copy()

        mask_experiments = ['', '_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                   '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb']

        # Preprocess Image
        # homomorphic_img, nml_homomorphic_img, multi_scale_img, img_adapteq, smoothed_img, fgf_image, filtered_fgf, multi_scale_img_only, enhanced_vessels
        preprocessed_imgs = {k: adjust_scale(v) for k, v in
                             zip(['_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                                  '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb'],
                                 preprocess(original_image)) if k in experiments}

        preprocessed_imgs[''] = original_image

        #make sure all iamges have 3 channels or cv2.merge
        preprocessed_imgs = {k: cv2.merge([v, v, v]) if len(v.shape) == 2 else v for k, v in preprocessed_imgs.items()}

        experiments_matrix = {}
        mask_matrix = {}
        for experiment, model in models.items():
            if experiment in mask_experiments:

                results = model.predict(preprocessed_imgs[experiment])[0]  # predict on an image

                experiments_matrix[experiment] = result_matrix_single_probability(results, prob_threshold=confidence)

                mask_matrix[experiment] = result_matrix_single(results)

        #create mask
        vessel_map_mask = create_mask_from_model_output(mask_matrix).astype(np.uint8)
        mask = skimage.morphology.binary_dilation(vessel_map_mask, skimage.morphology.disk(1))
        vessel_map_mask = erode_n(vessel_map_mask, 3)
        vessel_map_image = preprocessed_imgs['_bwn_ms_hist_gauss_fgf'].copy()[..., 0] * mask
        vessel_map_image = cv2.merge([vessel_map_image, vessel_map_image, vessel_map_image])

        #run model on new image mask
        model = models['_vessel_map_600epoch']
        results = model.predict(vessel_map_image, conf=confidence)[0]
        experiments_matrix['_vessel_map_600epoch'] = result_matrix_single_probability(results)

        #generate experiments for final matrix
        experiments_matrix = {k: v for k, v in experiments_matrix.items() if k in experiments}

        experiments_matrix["final_mask"] = np.zeros((27, 512, 512), np.float32)

        # create ensemble mask
        try:
            # iterate through masks and identify single class from each
            for exp, result_mask in experiments_matrix.items():
                if exp == "_merge":
                    pass
                else:
                    experiments_matrix["final_mask"] += result_mask

            final_mask = experiments_matrix["final_mask"]

            # 27, 512, 512
            n_exp_pred = (len([v.sum() for k, v in experiments_matrix.items() if v.sum() > 0]) - 1)  # number of models that predicted something

            # average prediction per pixel
            best_model_mask = final_mask / n_exp_pred

            # select best predicition class
            best_model_max_bool = best_model_mask.argmax(axis=0)
            best_mask = np.zeros((27, 512, 512))
            for ii in range(1, 27):
                best_mask[ii][best_model_max_bool == ii] = 1
            best_mask[best_mask == 1] = best_model_mask[best_mask == 1]

            # Filter wrong sided predictions
            num_5, num_1 = 0, 0
            for exp, result_mask in experiments_matrix.items():
                num_5 += (
                    1
                    if ("5" in [seg_dict[i - 1] for i in range(27) if result_mask[i].max() > 0]) | ("6" in [seg_dict[i - 1] for i in range(27) if result_mask[i].max() > 0])
                    else 0
                    )
                num_1 += (
                    1
                    if ("1" in [seg_dict[i - 1] for i in range(27) if result_mask[i].max() > 0]) | ("2" in [seg_dict[i - 1] for i in range(27) if result_mask[i].max() > 0])
                    else 0
                )

            print(f'num_5 {num_5} | num_1 {num_1}', best_mask.shape)

            if num_5 > num_1:
                category_list = [
                    i - 1 for i in range(27) if best_mask[i].max() > (1 / n_exp_pred)
                ]
                masks = [best_mask[i + 1] > (1 / n_exp_pred) for i in category_list]
            else:
                category_list = [i - 1 for i in range(27) if best_mask[i].max() > 0]
                masks = [best_mask[i + 1] > 0 for i in category_list]

            out_category_list = np.asarray([_ for _ in category_list])
            masks = np.array(masks)

            if num_5 > num_1:
                bad_indicies = list(np.where(np.isin(out_category_list, [0, 1, 2, 3, 19, 20, 21, 22], invert=True))[0])
                print('left side prediction: ', bad_indicies)
            else:
                bad_indicies = list(np.where(np.isin(out_category_list, [0, 1, 2, 3, 19, 20, 21, 22]))[0])
                print('right side prediction: ', bad_indicies)
            print(bad_indicies, out_category_list)

            out_category_list = out_category_list[bad_indicies]
            masks = np.array(masks)[np.array(bad_indicies)]

            # order segments from lowest to highest
            args_order = np.argsort(out_category_list)
            out_category_list = out_category_list[args_order]
            masks = masks[args_order, ...]

            # exclude segments that dont have enough training data
            new_masks = np.array(
                [
                    _
                    for i, _ in zip(out_category_list, masks)
                    if i + 1 not in exclude_segments.keys()
                ]
            )
            new_cat_list = np.array(
                [_ for _ in out_category_list if _ + 1 not in exclude_segments.keys()]
            )

            new_masks = np.array(new_masks)
            new_cat_list = np.array(new_cat_list)

            output_matrix[j - 1] = logical_constructor_ensemble(
                results=results, vessel_map=vessel_map_mask, category_list=new_cat_list, mask=new_masks
            )

            experiments_matrix["final_mask"] = output_matrix[j - 1]

        except Exception as e:
            print(e)
            output_matrix[j - 1] = experiments_matrix["_ms"]

        # exclude segments that dont have enough training data a second time
        for ii_ in range(1, 27):
            if ii_ in post_exclusion_segments.keys():
                output_matrix[j - 1, ii_] = np.zeros((512, 512))

        # over_ride segments
        if num_1 > num_5:
            # get additional info from best model _ms
            model_order_list = ["_bw", '_bwn_ms_hist_gauss', "_ddfb", "_ms"]
            for i in [0, 1, 2, 3]:
                for exp in model_order_list:
                    model_mask = experiments_matrix[exp]
                    if (model_mask[i + 1] > 0).sum() > 0:
                        output_matrix[j - 1, i + 1] = model_mask[i + 1] > 0
                        break
                    else:
                        output_matrix[j - 1, i + 1] = output_matrix[j - 1, i + 1] > 0
        elif num_1 == num_5:
            pass
        else:
            model_order_list = ["_bwn_ms_hist_gauss_fgf", "_ddfb", "_ms"]
            for i in [7]:
                for exp in model_order_list:
                    model_mask = experiments_matrix[exp]
                    if (model_mask[i + 1] > 0).sum() > 0:
                        output_matrix[j - 1, i + 1] = model_mask[i + 1] > 0
                        break
                    else:
                        output_matrix[j - 1, i + 1] = output_matrix[j - 1, i + 1] > 0

            model_order_list = ["_bwn", "_bwn_ms_hist_gauss_fgf", "_ddfb", "_ms"]
            for i in [4]:
                for exp in model_order_list:
                    model_mask = experiments_matrix[exp]
                    if (model_mask[i + 1] > 0).sum() > 0:
                        output_matrix[j - 1, i + 1] = model_mask[i + 1] > 0
                        break
                    else:
                        output_matrix[j - 1, i + 1] = output_matrix[j - 1, i + 1] > 0

        ####Using Post Construction Logic

        seed_segment = '5' if (num_5 > num_1) | (output_matrix[j-1, 6].sum() > 0) else '1'
        print(f"trying post connection logic with seed: {seed_segment}")
        output_matrix[j-1] = post_connection_engine(root_seg=seed_segment, output_matrix=output_matrix[j-1], vessel_map_mask=vessel_map_mask)

        print('predicted classes:\t', [seg_dict[i - 1] for i in range(1, 27) if output_matrix[j-1, i].max() > 0])

        ## apply vesselmap to each layer
        for i in range(27):
            output_matrix[j - 1, i] = output_matrix[j - 1, i] * vessel_map_mask

        # size filter
        output_matrix = size_filter(output_matrix, j)

    # create json
    output_json = export_json(output_matrix, output_json_path=output_json_path, num_images=num_images, mapping_dict=inverse_dict)

    return output_json

