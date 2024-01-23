import torch
from tqdm import tqdm
from PIL import Image
import OpenEXR as exr
import Imath
import os
import numpy as np
from torchvision.datasets import Cityscapes

from definitions import *

class DatasetMiniscapes(torch.utils.data.Dataset):
    '''
    This class read the Synscapes dataset
    '''

    def __init__(self, dataset_root, split_idx: list, integrity_check=False):
        '''
        dataset_root: root folder whose sub-folders are <class>,<depth>,<rgb>,.. so on
        split_idx: indices corresponding to the current dataset
        '''
        self.dataset_root = dataset_root
        self.split_idx = split_idx
        self.transforms = None
        if integrity_check:
            for i in tqdm(range(len(self))):
                self.get(i)

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        # load rgb, Image.open does not load contents of file in memory, it checks minimally
        rgb = Image.open(self.get_item_path(index, MOD_RGB))
        # Image.load() put the contents of file in memory
        rgb.load()
        assert rgb.mode == 'RGB'

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
        }

        # load semseg
        path_semseg = self.get_item_path(index, MOD_SEMSEG)
        if os.path.isfile(path_semseg):
            semseg = self.load_semseg(path_semseg)
            assert semseg.size == rgb.size
            out[MOD_SEMSEG] = semseg

        # load depth
        path_depth = self.get_item_path(index, MOD_DEPTH)
        if os.path.isfile(path_depth):
            # the depth map here is a np array object
            # we need to convert it to PIL object since transforms later on requires PIL object
            depth = self.load_exr(path_depth)
            disparity = self.depth_meters_float32_to_disparity_uint8(depth, out_of_range_policy='clamp_to_range')
            disparity_im = Image.fromarray(disparity)
            # assert depth.size == rgb.size
            out[MOD_DEPTH] = disparity_im  # depth
            # out['disparity'] = disparity_im

        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            #print(f"index:{index}")
            out = self.transforms(out)

        return out

    def __getitem__(self, index):
        # index into the stored index in the object
        index_ = self.split_idx[index]
        return self.get(index_)

    def __len__(self):
        return len(self.split_idx)

    def get_item_path(self, index, modality):
        if modality == MOD_SEMSEG:
            modality = modality + "_filt"
            return self.dataset_root / modality / f'{index}{".exr" if modality == MOD_DEPTH else ".png"}'
        return self.dataset_root / modality / f'{index}{".exr" if modality == MOD_DEPTH else ".png"}'

    def name_from_index(self, index):
        return f'{index}'

    @property
    def rgb_mean(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        return [255 * 0.485, 255 * 0.456, 255 * 0.406]

    @property
    def rgb_stddev(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        return [255 * 0.229, 255 * 0.224, 255 * 0.225]

    @staticmethod
    def load_semseg(filepath):
        semseg = Image.open(filepath)
        # L (8-bit pixels, grayscale), P (8-bit pixels, mapped to any other mode using a color palette)
        assert semseg.mode in ('P', 'L')
        return semseg

    @staticmethod
    def load_exr(filepath):
        exrfile = exr.InputFile(filepath)
        header = exrfile.header()
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        channelData = dict()
        # convert all channels in the image to numpy arrays
        for c in header['channels']:
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.frombuffer(C, dtype=np.float32)
            C = np.reshape(C, isize)
            channelData[c] = C

        Z = None if 'Z' not in header['channels'] else channelData['Z']
        return Z

    def depth_meters_float32_to_disparity_uint8(self, x, out_of_range_policy):
        assert out_of_range_policy in ('invalidate', 'clamp_to_range')
        x = np.array(x).astype(np.float32)
        # compute disparity from depth
        x = 1 / x
        disparity_min, disparity_max = 1 / self.depth_meters_max, 1 / self.depth_meters_min
        x = 1 + 254 * (x - disparity_min) / (disparity_max - disparity_min)
        if out_of_range_policy == 'invalidate':
            with np.errstate(invalid='ignore'):
                x[x < 0.] = float('nan')
                x[x > 255.] = float('nan')
            x[x != x] = 0
        elif out_of_range_policy == 'clamp_to_range':
            assert np.sum((x != x).astype(np.int32)) == 0
            x[x < 1.] = 1.
            x[x > 255.] = 255.
        x = x.astype(np.uint8)
        return x

    def depth_disparity_uint8_to_meters_float32(self, x, check_all_pixels_valid):
        assert type(check_all_pixels_valid) is bool
        mask_invalid = x == 0
        assert not check_all_pixels_valid or np.sum(mask_invalid.astype(np.int32)) == 0
        disparity_min, disparity_max = 1 / self.depth_meters_max, 1 / self.depth_meters_min
        x = (disparity_max - disparity_min) * (x - 1).astype(np.float32) / 254 + disparity_min
        x = 1.0 / x
        x[mask_invalid] = 0.0
        return x

    def save_depth(self, path, img, out_of_range_policy):
        assert torch.is_tensor(img) and (img.dim() == 2 or img.dim() == 3 and img.shape[0] == 1)
        if img.dim() == 3:
            img = img.squeeze(0)
        img = img.cpu().numpy()
        img = self.depth_meters_float32_to_disparity_uint8(img, out_of_range_policy)
        img = Image.fromarray(img)
        img.save(path, optimize=True)

    @staticmethod
    def save_semseg(path, img, semseg_color_map, semseg_ignore_label=None, semseg_ignore_color=(0, 0, 0)):
        if torch.is_tensor(img):
            img = img.squeeze()
            assert img.dim() == 2 and img.dtype in (torch.int, torch.long)
            img = img.cpu().byte().numpy()
            img = Image.fromarray(img, mode='P')
        palette = [0 for _ in range(256 * 3)]
        for i, rgb in enumerate(semseg_color_map):
            for c in range(3):
                palette[3 * i + c] = rgb[c]
        if semseg_ignore_label is not None:
            for c in range(3):
                palette[3 * semseg_ignore_label + c] = semseg_ignore_color[c]
        img.putpalette(palette)
        img.save(path, optimize=True)

    @property
    def semseg_num_classes(self):
        return len(self.semseg_class_names)

    @property
    def semseg_class_names(self):
        return [clsdesc.name for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]

    @property
    def semseg_class_colors(self):
        return [clsdesc.color for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]

    @property
    def semseg_ignore_label(self):
        return 255

    @property
    def depth_meters_mean(self):
        return 27.0727

    @property
    def depth_meters_stddev(self):
        return 29.1264

    @property
    def depth_meters_min(self):
        return 4

    @property
    def depth_meters_max(self):
        return 300