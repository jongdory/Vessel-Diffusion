import os
import numpy as np
import argparse
import SimpleITK as sitk
import torch
import torch.quantization as quant
from pytorch_lightning.trainer import Trainer
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from torch.cuda.amp import GradScaler, autocast
from typing import Union, Tuple, List

from omegaconf import OmegaConf
from ddpm.util import instantiate_from_config
from ddpm.models.diffusion.ddim import DDIMSampler
from monai.transforms.utils_pytorch_numpy_unification import percentile

from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_dilation, zoom

def save_label(mask, label_arr, save_path, affine, header):

    kernel = np.ones((5,5,5), np.uint8)
    label_arr = binary_dilation(binary_erosion(label_arr, structure=kernel), structure=kernel)

    masses = []

    # devide mass
    labeledImage = label(mask!=0)

    # get mass size
    areas = [r.area for r in regionprops(labeledImage)]
    sorted_areas = sorted(areas, reverse=True)
    
    for area in sorted_areas:
        if area < 10000: break
        index = areas.index(area) + 1
        zeros = np.zeros_like(mask)
        zeros[labeledImage==index] = 1
        # masses.append(areas.index(area) + 1)
        if np.sum(zeros * label_arr) > 0:
            masses.append(areas.index(area) + 1)
            
    mask = np.zeros_like(mask)
    for massIndex in masses:
        mask[labeledImage==massIndex] = 1

    nii = nib.Nifti1Image(mask, affine, header)
    nib.save(nii, save_path)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="preds",
    )
    parser.add_argument(
        "-st",
        "--start_time",
        type=int,
        default=501,
    )
    return parser

def maybe_mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


def fuse_modules_recursively(module, prefix=""):
    for sub_module_name, sub_module in module.named_children():
        if sub_module_name == "time_embed" or sub_module_name == "time_emb_proj":
            continue
        full_name = f"{prefix}.{sub_module_name}" if prefix else sub_module_name
        if not sub_module.named_children():
            quant.fuse_modules(sub_module, [[sub_module_name]], inplace=True)
        else:
            fuse_modules_recursively(sub_module, full_name)


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def sigmoid(x):
    return 1 / (1 +np.exp(-x))


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit
    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


class Predictor():
    def __init__(self, model, start_time):

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        self.net = model
        self.net.learning_rate = 0.0002
        self.net.register_schedule(linear_start=0.0015, linear_end=0.0195, timesteps=1000)
        # self.net.instantiate_cond_stage(self.net.cond_stage_config)
        self.ddim_sampler = DDIMSampler(self.net)
        self.ddim_sampler.make_schedule(ddim_num_steps=50)
        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channels we have in the output. Important for preallocation in inference
        self.num_classes = 1  # number of channels in the output

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None
        self.start_time = start_time


    def __call__(self, dataloader, savepath):
        torch.cuda.empty_cache()

        maybe_mkdir(savepath)
        for item in dataloader._test_dataloader():
            subject_id = item[f"image_meta_dict"]['filename_or_obj'][0].split('/')[-2]
            # subject_id = item[f"mask_meta_dict"]['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]
            print(subject_id, " processing...")
            img = nib.load(item[f"image_meta_dict"]['filename_or_obj'][0])

            data, data_cond, label = item['image'].squeeze(axis=0).cuda(), item['frangi'].squeeze(axis=0).cuda(), item['mask'].squeeze(axis=0).cuda()
            data = torch.cat((data, data_cond), dim=0)
            
            patch_size = (160,160,160)
            steps = self._compute_steps_for_sliding_window(patch_size, tuple(data.shape[1:]), 1)
            
            steps = [step if step != [] else [0] for step in steps ]
            
            segs, prob = self._get_segmap(data, label, patch_size, steps)

            data = data.detach().cpu().numpy()[0]
            data = data[::-1,:,::-1]
            nib.save(nib.Nifti1Image(data, img.affine, img.header), f"{savepath}/{subject_id}_data.nii.gz")

            segs = segs[::-1,:,::-1]
            nib.save(nib.Nifti1Image(segs, img.affine, img.header), f"{savepath}/{subject_id}-pred.nii.gz")

            prob = prob[::-1,:,::-1]
            nib.save(nib.Nifti1Image(prob, img.affine, img.header), f"{savepath}/{subject_id}_prob.nii.gz")

            label = label.detach().cpu().numpy()[0]
            label = label[::-1,:,::-1]
            nib.save(nib.Nifti1Image(label, img.affine, img.header), f"{savepath}/{subject_id}_label.nii.gz")

            save_label(segs, label , f"{savepath}/{subject_id}_mask.nii.gz", img.affine, img.header)


    def get_device(self):
        if next(self.net.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.net.parameters()).device.index


    def _get_segmap(self, data, label, patch_size, steps):

        data, slicer = pad_nd_image(data.cpu().numpy(), patch_size, "constant", None, True, None)

        mirror_axes = (0, 1) #(0, 1, 2) # (0, 1)
        aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], 
                        label[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], 
                        mirror_axes, True)[0]

                    predicted_patch = predicted_patch.detach().cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch

        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
            range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])

        aggregated_results = aggregated_results[slicer]
        predicted_segmentation = np.zeros(aggregated_results.shape[1:])
        predicted_segmentation[aggregated_results[0]  > 0.4] = 1

        return predicted_segmentation, aggregated_results[0]
    
    # skip_timestep!
    def _inference_dm(self, x, label, flip):
        with autocast(enabled=True):            
            if flip is not None:
                x, label = torch.flip(x, flip), torch.flip(label, flip)

            shape = label.shape[0:]
            self.net = self.net.cuda()
            cond = self.net.get_learned_conditioning(x)
            skip_timesteps = self.start_time
            samples, intermediates = self.ddim_sampler.ddim_sampling(cond=cond, shape=shape, x0=label, skip_timesteps=skip_timesteps)
            samples[samples < 0] = 0

        return samples

    def scale_norm(self, img):
        lower=0
        upper=100
        b_min=0
        b_max=1
        a_min = percentile(img, lower)  # type: ignore
        a_max = percentile(img, upper)  # type: ignore
        
        if a_max - a_min == 0.0:
            if b_min is None:
                return img - a_min
            return img - a_min + b_min

        img = (img - a_min) / (a_max - a_min)
        if (b_min is not None) and (b_max is not None):
            img = img * (b_max - b_min) + b_min
        
        return img


    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor],
                                           label: Union[np.ndarray, torch.tensor],
                                            mirror_axes: tuple,
                                            do_mirroring: bool = True,
                                            mult: np.ndarray or torch.tensor = None) -> torch.tensor:
            assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

            # if cuda available:
            #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
            #   we now return a cuda tensor! Not numpy array!

            x = maybe_to_torch(x)
            result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                    dtype=torch.float)
            
            x = self.scale_norm(x)

            if torch.cuda.is_available():
                x = x.cuda()
                result_torch = result_torch.cuda()

            if mult is not None:
                mult = maybe_to_torch(mult)
                if torch.cuda.is_available():
                    mult = mult.cuda()

            if do_mirroring:
                mirror_idx = 8
                num_results = 2 ** len(mirror_axes)
            else:
                mirror_idx = 1
                num_results = 1

            for m in range(mirror_idx):
                if m == 0:
                    pred = self._inference_dm(x, label, None)
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    # continue
                    pred = self._inference_dm(x, label, (4, ))
                    result_torch += 1 / num_results * torch.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self._inference_dm(x, label, (3, ))
                    result_torch += 1 / num_results * torch.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    # continue
                    pred = self._inference_dm(x, label, (4, 3))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self._inference_dm(x, label, (2, ))
                    result_torch += 1 / num_results * torch.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    # continue
                    pred = self._inference_dm(x, label, (4, 2))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    # continue
                    pred = self._inference_dm(x, label, (3, 2))
                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))
                
                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self._inference_dm(x, label, (4, 3, 2))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))
            
            if mult is not None:
                result_torch[:, :] *= mult

            return result_torch


    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map


    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps



def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model = instantiate_from_config(config.model).cuda()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    predictor = Predictor(model, opt.start_time)
    pred_path = opt.outdir
    print(pred_path)
    maybe_mkdir(pred_path)
    predictor(data, pred_path)

if __name__ == "__main__":
    main()