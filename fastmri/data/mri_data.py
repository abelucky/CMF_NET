
import csv
import os

import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from os.path import splitext
from os import listdir, path
import cv2
import scipy.io as sio
from fastmri.data import transforms
from scipy.io import loadmat

def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/chunmeifeng/Data/",
            brain_path="/home/chunmeifeng/Data/",
            # log_path="/home/chunmeifeng/experimental/MINet/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            challenge,
            sample_rate=1,
            dataset_cache_file=pathlib.Path("dataset_cache.pkl"),
            num_cols=None,
            mode='train',
    ):
        self.mode = mode

        #challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        #transform
        self.transform = transform

        self.examples=[]

        self.cur_path=root
        self.csv_file=os.path.join(self.cur_path,"singlecoil_"+self.mode+"_split_less.csv")

        #读取CSV
        with open(self.csv_file,'r') as f:
            reader=csv.reader(f)

            for row in reader:
                pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path,row[0]+'.h5'))

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1]+'.h5'))

                for slice_id in range(min(pd_num_slices,pdfs_num_slices)):
                    self.examples.append((os.path.join(self.cur_path, row[0]+'.h5'),os.path.join(self.cur_path, row[1]+'.h5')
                                          ,slice_id,pd_metadata,pdfs_metadata))

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples=self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        #读取pd
        pd_fname,pdfs_fname,slice,pd_metadata,pdfs_metadata = self.examples[i]

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]

            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pd_metadata)

        if self.transform is None:
            pd_sample = (pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)
        else:
            pd_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)

        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][slice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pdfs_metadata)

        if self.transform is None:
            pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        else:
            pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)


        return (pd_sample,pdfs_sample)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices


class IXIdataset(Dataset):
    def __init__(self, root, challenge, mode='train'):
        self.data_dir = root
        self.num_input_slices = 1
        self.img_size = 224
        self.mode = mode

        self.examples = []
        self.cur_path = root
        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")
        # 读取CSV
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
        #with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                #for slice_id in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]:
                for slice_id in range(20, 120):
                #for slice_id in [18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94]:
                    self.examples.append((row[0],  row[1], slice_id))


    def __len__(self):
        return len(self.examples)

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def niipath2matpath(self, T1,slice_id):
        filedir,filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir,'mat1')
        basename, ext = path.splitext(filename)
        base_name = basename[:-1]
        file_name = '%s-%03d.mat'%(base_name,slice_id)
        T1_file_path = path.join(mat_dir,file_name)
        return T1_file_path

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
        img = img[:, None, :, :]
        return img

    # @classmethod
    def slice_preprocess(self, kspace_cplx):  # 256,256
        # crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # split to real and imaginary channels
        kspace = np.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
        # target image:
        image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        kspace = kspace.transpose((2, 0, 1))  # 2,256,256
        # print('kspace:',kspace.shape)
        HR = self.getHR(image)
        LR, LR_ori = self.getLR(image)

        return LR, LR_ori, kspace, HR

    def getHR(self, hr_data):
        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (224, 224))
        imgifft = np.fft.ifft2(imgfft)
        img_out = abs(imgifft)

        return img_out

    def getLR(self, hr_data):
        # imgfft = np.fft.fft2(hr_data)
        #
        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (56, 56))

        # masks_dictionary = loadmat(
        #     "/opt/data/private/re_sr/experimental/CMFNet/1D-Cartesian_6X_112112.mat")
        # mask = masks_dictionary['mask']

        mask = np.load("/opt/data/private/re_sr/experimental/CMFNet/random45656.npy")

        t = imgfft

        imgfft = imgfft * mask

        imgifft = np.fft.ifft2(imgfft)
        img_out = abs(imgifft)

        t = np.fft.ifft2(t)
        LR_ori = abs(t)

        return img_out, LR_ori

    def __getitem__(self, i):

        #fname, slice_num = self.ids[i]
        T1_fname, T2_fname, slice_num = self.examples[i]
        # target_LR_T2=np.zeros((1, 64, 64))
        # target_img_T2 = np.zeros((1, self.img_size, self.img_size))

        slice_range = [slice_num]

        # 读取T1
        for ii, slice_id in enumerate(slice_range):
            full_file_path = path.join(self.data_dir, T1_fname + '-{:03d}.mat'.format(slice_id))
            full_file_path = full_file_path.replace('/h5/', '/mat/')

            img_T1 = sio.loadmat(full_file_path)['img']

            img_T1, T1_mean, T1_std = transforms.normalize_instance(img_T1, eps=1e-11)

            img_T1_height, img_T1_width = img_T1.shape
            img_T1_matRotate = cv2.getRotationMatrix2D((img_T1_height * 0.5, img_T1_width * 0.5), 90, 1)
            img_T1 = cv2.warpAffine(img_T1, img_T1_matRotate, (img_T1_height, img_T1_width))

            kspace_T1 = self.fft2(img_T1)  # ksapce: (256,256)
            slice_LR_T1,slice_LR_T1_ori, slice_full_Kspace_T1, slice_full_img_T1 = self.slice_preprocess(kspace_T1)

            if slice_id == slice_num:
                target_LR_T1 = slice_LR_T1
                target_img_T1 = slice_full_img_T1
                target_LR_T2_ori = slice_LR_T1_ori

                break
        T1_LR_image = torch.from_numpy(target_LR_T1)
        T1_target = torch.from_numpy(target_img_T1)
        T1_LR_ori = torch.from_numpy(target_LR_T2_ori)

        T1_sample = (T1_LR_image, T1_LR_ori, T1_mean, T1_std, T1_fname, slice_num)

        # 读取T2
        for ii, slice_id in enumerate(slice_range):
            full_file_path = path.join(self.data_dir, T2_fname + '-{:03d}.mat'.format(slice_id))
            full_file_path = full_file_path.replace('/h5/', '/mat/')

            img_T2 = sio.loadmat(full_file_path)['img']

            img_T2, T2_mean, T2_std = transforms.normalize_instance(img_T2, eps=1e-11)

            img_T2_height, img_T2_width = img_T2.shape
            img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
            img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))

            kspace_T2 = self.fft2(img_T2)  # ksapce: (256,256)
            slice_LR_T2,slice_LR_T2_ori, slice_full_Kspace_T2, slice_full_img_T2 = self.slice_preprocess(kspace_T2)

            if slice_id == slice_num:
                target_LR_T2 = slice_LR_T2
                target_LR_T2_ori = slice_LR_T2_ori
                target_img_T2 = slice_full_img_T2

                break
        T2_LR_image = torch.from_numpy(target_LR_T2)
        T2_target = torch.from_numpy(target_img_T2)
        T2_LR_ori = torch.from_numpy(target_LR_T2_ori)

        T2_sample = (T2_LR_image, T2_target, T2_mean, T2_std, T2_fname, slice_num, T2_LR_ori)


        return (T1_sample, T2_sample)

    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        # print(data.shape)
        # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
        assert 0 < shape[1] <= data.shape[-1]  # 640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]

    def contrastStretching(self,img, saturated_pixel=0.004):
        """ constrast stretching according to imageJ
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
        values = np.sort(img, axis=None)
        nr_pixels = np.size(values)  # 像素数目
        lim = int(np.round(saturated_pixel*nr_pixels))
        v_min = values[lim]
        v_max = values[-lim-1]
        img = (img - v_min)*(255.0)/(v_max - v_min)
        img = np.minimum(255.0, np.maximum(0.0, img))  # 限制到0-255区间
        return img

# class IXIdataset(Dataset):
#     def __init__(self, root, challenge, mode='train'):
#         self.data_dir = root
#         self.num_input_slices = 1
#         self.img_size = 256
#         self.mode = mode
#
#         self.examples = []
#         self.cur_path = root
#         self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")
#         # 读取CSV
#         with open(self.csv_file, 'r') as f:
#             reader = csv.reader(f)
#
#             for row in reader:
#                 for slice_id in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]:
#                     self.examples.append((row[0], slice_id))
#
#
#     def __len__(self):
#         return len(self.examples)
#
#     def crop_toshape(self, kspace_cplx):
#         if kspace_cplx.shape[0] == self.img_size:
#             return kspace_cplx
#         if kspace_cplx.shape[0] % 2 == 1:
#             kspace_cplx = kspace_cplx[:-1, :-1]
#         crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
#         kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
#         return kspace_cplx
#
#     def niipath2matpath(self, T1,slice_id):
#         filedir,filename = path.split(T1)
#         filedir,_ = path.split(filedir)
#         mat_dir = path.join(filedir,'mat1')
#         basename, ext = path.splitext(filename)
#         base_name = basename[:-1]
#         file_name = '%s-%03d.mat'%(base_name,slice_id)
#         T1_file_path = path.join(mat_dir,file_name)
#         return T1_file_path
#
#     def ifft2(self, kspace_cplx):
#         return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]
#
#     def fft2(self, img):
#         return np.fft.fftshift(np.fft.fft2(img))
#
#     def inverseFT(self, Kspace):
#         Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
#         img_cmplx = torch.ifft(Kspace, 2)
#         img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
#         img = img[:, None, :, :]
#         return img
#
#     # @classmethod
#     def slice_preprocess(self, kspace_cplx):  # 256,256
#         # crop to fix size
#         kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
#         # split to real and imaginary channels
#         kspace = np.zeros((self.img_size, self.img_size, 2))  # 256,256,2
#         kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
#         kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
#         # target image:
#         image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
#         # HWC to CHW
#         kspace = kspace.transpose((2, 0, 1))  # 2,256,256
#         # print('kspace:',kspace.shape)
#         HR = self.getHR(image)
#         LR, LR_ori = self.getLR(image)
#
#         return LR, LR_ori, kspace, HR
#
#     def getHR(self, hr_data):
#         imgfft = self.fft2(hr_data)
#
#         imgfft = self.center_crop(imgfft, (256, 256))
#         imgifft = np.fft.ifft2(imgfft)
#         img_out = abs(imgifft)
#
#         return img_out
#
#     def getLR(self, hr_data):
#         # imgfft = np.fft.fft2(hr_data)
#         #
#         imgfft = self.fft2(hr_data)
#
#         imgfft = self.center_crop(imgfft, (128, 128))
#
#         masks_dictionary = loadmat(
#             "/opt/data/private/re_sr/experimental/MINet/1D-Cartesian_6X_128128.mat")
#         mask = masks_dictionary['mask']
#
#         t = imgfft
#
#         imgfft = imgfft * mask
#
#         imgifft = np.fft.ifft2(imgfft)
#         img_out = abs(imgifft)
#
#         t = np.fft.ifft2(t)
#         LR_ori = abs(t)
#
#         return img_out, LR_ori
#
#     def __getitem__(self, i):
#
#         #fname, slice_num = self.ids[i]
#         T2_fname, slice_num = self.examples[i]
#         target_LR_T2=np.zeros((1, 128, 128))
#         target_img_T2 = np.zeros((1, self.img_size, self.img_size))
#
#         slice_range = [slice_num]
#
#         # 读取T2
#         for ii, slice_id in enumerate(slice_range):
#             full_file_path = path.join(self.data_dir, T2_fname + '-{:03d}.mat'.format(slice_id))
#             full_file_path = full_file_path.replace('/h5/', '/mat/')
#
#             img_T2 = sio.loadmat(full_file_path)['img']
#
#             img_T2, T2_mean, T2_std = transforms.normalize_instance(img_T2, eps=1e-11)
#
#             img_T2_height, img_T2_width = img_T2.shape
#             img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
#             img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))
#
#             kspace_T2 = self.fft2(img_T2)  # ksapce: (256,256)
#             slice_LR_T2,slice_LR_T2_ori, slice_full_Kspace_T2, slice_full_img_T2 = self.slice_preprocess(kspace_T2)
#
#             if slice_id == slice_num:
#                 target_LR_T2 = slice_LR_T2
#                 target_LR_T2_ori = slice_LR_T2_ori
#                 target_img_T2 = slice_full_img_T2
#
#                 break
#         T2_LR_image = torch.from_numpy(target_LR_T2)
#         T2_target = torch.from_numpy(target_img_T2)
#         T2_LR_ori = torch.from_numpy(target_LR_T2_ori)
#
#         T2_sample = (T2_LR_image, T2_target, T2_mean, T2_std, T2_fname, slice_num, T2_LR_ori)
#
#
#         return  T2_sample
#
#     def center_crop(self, data, shape):
#         """
#         Apply a center crop to the input real image or batch of real images.
#
#         Args:
#             data (torch.Tensor): The input tensor to be center cropped. It should have at
#                 least 2 dimensions and the cropping is applied along the last two dimensions.
#             shape (int, int): The output shape. The shape should be smaller than the
#                 corresponding dimensions of data.
#
#         Returns:
#             torch.Tensor: The center cropped image
#         """
#         # print(data.shape)
#         # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
#         assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
#         assert 0 < shape[1] <= data.shape[-1]  # 640...640
#         w_from = (data.shape[-2] - shape[0]) // 2
#         h_from = (data.shape[-1] - shape[1]) // 2
#         w_to = w_from + shape[0]
#         h_to = h_from + shape[1]
#         return data[..., w_from:w_to, h_from:h_to]
#
#     def contrastStretching(self,img, saturated_pixel=0.004):
#         """ constrast stretching according to imageJ
#         http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
#         values = np.sort(img, axis=None)
#         nr_pixels = np.size(values)  # 像素数目
#         lim = int(np.round(saturated_pixel*nr_pixels))
#         v_min = values[lim]
#         v_max = values[-lim-1]
#         img = (img - v_min)*(255.0)/(v_max - v_min)
#         img = np.minimum(255.0, np.maximum(0.0, img))  # 限制到0-255区间
#         return img