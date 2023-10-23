import gc
import torch
from ultralytics import YOLO

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    pass

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import time

from utils.preprocess import *
from arcade_challenge_utils.load_json_annotations import *
from arcade_challenge_utils.utils import *

# import deepcopy
from PIL import Image
import skimage

disk_size = 5

if __name__ == '__main__':
    YOLO_VERBOSE = False
    # Garbage Collect
    gc.collect();
    torch.cuda.empty_cache()

    arcade = Path('/data2/arcade_challenge/arcade_dataset_phase_1_with_val')

    alpha = 0.

    new_experiment_name = 'vessel_map_large_disks'
    single_or_multi = 'multi'
    epochs = 600
    disk_size = 45
    
    for phase in ['train', 'val']:

        for dataset in ['segmentation', 'stenosis']:

            short_dataset = dataset[:4] if 'stenosis' in dataset else dataset[:3]

            shutil.copy('/home/txl4827/projects/ARCADE/useful scripts/empty_annotations.json',
                        f'/data2/arcade_challenge/arcade_dataset_phase_1_with_val/{dataset}_dataset/')

            experiments = ['', '_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                           '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb']

            experiments_matrix = {}

            num_images = 200 if 'val' in phase else 1000

            empty_json = '/home/txl4827/projects/ARCADE/useful scripts/empty_annotations.json'
            empty_json_path = f"/data2/arcade_challenge/arcade_dataset_phase_1_with_val/{dataset}_dataset/empty_annotations.json"

            try:
                os.makedirs(arcade / f'{dataset}_dataset_{new_experiment_name}/{phase}/images')
            except:
                pass

            models = {}

            gt_mask = load_annotations(arcade / f'{dataset}_dataset/train/annotations/train.json', num_imgs=1000)

            try:
                os.makedirs(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask/')
            except:
                pass


            #Load models
            output_matrix = np.zeros((num_images, 27, 512, 512), np.int32)
            for experiment in experiments:
                model_path = f"/home/txl4827/projects/syntax_score/arcade_challenge/segmentation_dataset{experiment}_{single_or_multi}_gamma_{epochs}epochs_0.5dropout/weights/best.pt"

                # Load a model
                # model = YOLO('models/yolov8n-seg.pt')  # load an official model
                model = YOLO(model_path)  # load a custom model
                # Predict with the model

                models[experiment] = model

            #run models on images to get masks
            for j in range(1, num_images+1):

                s = time.time()

                image_path = arcade / f'{dataset}_dataset/{phase}/images/{j}.png'
                # Load original image
                original_image = np.array(Image.open(image_path)).copy()

                # Preprocess Image
                #homomorphic_img, nml_homomorphic_img, multi_scale_img, img_adapteq, smoothed_img, fgf_image, filtered_fgf, multi_scale_img_only, enhanced_vessels
                preprocessed_imgs = {k: adjust_scale(v) for k, v in zip(['_bw', '_bwn', '_bwn_ms', '_bwn_ms_hist', '_bwn_ms_hist_gauss',
                                                           '_bwn_ms_hist_gauss_fgf', '_bwn_ms_hist_gauss_fgf_avg', '_ms', '_ddfb'],
                                                          preprocess(original_image)) if k in experiments}


                preprocessed_imgs[''] = original_image
                print('preprocessing time: ', time.time() - s)

                experiments_matrix = {}
                for experiment, model in models.items():
                    image_path = arcade / f'{dataset}_dataset{experiment}/{phase}/images/{j}.png'

                    results = model.predict(image_path)[0]  # predict on an image

                    experiments_matrix[experiment] = result_matrix_single(results)


                experiments_matrix['final_mask'] = np.zeros((27, 512, 512), np.uint8)

                print('prediction time: ', time.time() - s)

                for exp, result_mask in experiments_matrix.items():
                    if exp == '_merge':
                        pass
                    else:
                        experiments_matrix['final_mask'] = (experiments_matrix['final_mask'] == 1) | (result_mask == 1)

                        output_single = []
                        counter = 0
                        for i in result_mask:
                            if i.sum() > 0:
                                counter += 1

                        output_single.append(counter)
                        p(f'segments detected: {exp}', output_single)

                # create ensemble mask
                ensemble_mask = np.zeros((len(experiments_matrix.keys()), 27, 512, 512))

                # iterate through masks and identify single class from each
                for ii, exp in enumerate(experiments_matrix.keys()):
                    p(f'{ii + 1} of {len(experiments_matrix.keys())}')
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

                output_dir = arcade / f'{dataset}_dataset_{new_experiment_name}/{phase}/images'

                # create mask
                mask = (new_output_ensemble > 0).max(axis=0)
                mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(disk_size))

                #save mask to file
                try:
                    os.makedirs(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask/{phase}')
                except:
                    pass
                cv2.imwrite(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask/{phase}/{j}.png', mask.astype(np.uint8) * 255)

                output = preprocessed_imgs['_bwn_ms_hist_gauss_fgf'].copy()
                output[~mask] = 0 #255 - ((255 - output[~mask]) * alpha)

                multi_channel_output = output
                # multi_channel_output = []
                # for img in [output.astype(np.uint8), preprocessed_imgs['_bwn_ms_hist_gauss'].astype(np.uint8), preprocessed_imgs[''].astype(np.uint8)]:
                #     if len(img.shape) > 2:
                #         print(img.shape)
                #         img = img[:, :, 0]
                #     multi_channel_output.append(img)
                #
                # multi_channel_output = cv2.merge(multi_channel_output)

                # create final mask and apply to images
                output_dir = arcade / f'{dataset}_dataset_{new_experiment_name}/{phase}/images'
                # save image
                cv2.imwrite(str(output_dir / f'{j}.png'), multi_channel_output)

                print(f'{j} of {num_images} | time: {time.time() - s:.2f} | segments detected: {output_single}')

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(original_image, cmap='gray')
                axes[0].set_title('original')
                axes[1].imshow(output, cmap='gray')
                axes[1].set_title('final mask')
                axes[2].imshow(gt_mask[j-1].max(axis=0), cmap='gray')
                axes[2].set_title('ground truth')
                axes[3].imshow(new_output_ensemble.max(axis=0), cmap='gray')
                axes[3].set_title('predicted mask')
                plt.tight_layout()
                #set title of entire plot
                plt.suptitle(f'Image {j}')

                try:
                    plt.savefig(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask_validation/{j}.png')
                except:
                    os.makedirs(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask_validation/')
                    plt.savefig(f'/data2/arcade_challenge/{dataset}_{new_experiment_name}_mask_validation/{j}.png')

                plt.close('all')

