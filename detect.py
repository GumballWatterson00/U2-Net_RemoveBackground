import os
import errno
import time
import logging

import numpy as np
from PIL import Image
from skimage import io, transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms  # , utils

from u2net import model, utils
import libs.postprocessing as postprocessing

logger = logging.getLogger(__name__)
post = postprocessing.run()
basedir = os.path.abspath(os.path.dirname(__file__))


def run():
    return U2NET()


class U2NET: 
    
    def __init__(self):
        self.torch = torch
        self.Variable = Variable
        self.predicted = None
        self.item = None
        self.net = None
        self.org_item = None

    def load_model(self, model_name: str = "u2net"):
        net = model.U2NET(3, 1)
        try:
            if torch.cuda.is_available():
                filepath = os.path.join(basedir, (model_name + '.pth'))
                net.load_state_dict(torch.load(filepath))
                net.to(torch.device("cuda"))
            else:
                filepath = os.path.join(basedir, (model_name + '.pth'))
                net.load_state_dict(torch.load(filepath, map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath
            )

        net.eval()
        return net

    def norm_pred(self, predicted):
        ma = torch.max(predicted)
        mi = torch.min(predicted)
        out= (predicted - mi) / (ma - mi)

        return out

    def process(self, net, item):
        #Removes background from image and returns PIL RGBA Image.
        if isinstance(item, str):
            logger.debug("Load image: {}".format(item))
        image, org_image = self.load_image(item)  # Load image
        if image is False or org_image is False:
            return False
        image = self.predict(net, image, org_image)  
        image = post.run(self, image, org_image)
        return image

    def load_image(self, item):
        image_size = 320
        if isinstance(item, str):
            try:
                image = io.imread(item)
            except IOError:
                logger.error('Cannot retrieve image.')
                return False, False
            pil_image = Image.fromarray(image)
        else:
            image = np.array(item)
            pil_image = item
        image = transform.resize(image, (image_size, image_size), mode='constant')
        image = self.ndrarray2tensor(image)
        return image, pil_image

    def ndrarray2tensor(self, item: np.ndarray):
        tmp_img = np.zeros((item.shape[0], item.shape[1], 3))
        item /= np.max(item)
        if item.shape[2] == 1:
            tmp_img[:, :, 0] = (item[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (item[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (item[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (item[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (item[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (item[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return torch.from_numpy(tmp_img)

    def predict(self, net, item, org_item):
        
        image = item.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        mask, d2, d3, d4, d5, d6, d7 = net(image) #Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.norm_pred(mask)
        # Prepare mask
        logger.debug('Prepare mask')
        mask = mask.squeeze()
        mask_np = mask.cpu().detach().numpy()
        mask = Image.fromarray(mask_np * 255).convert("L")
        mask = mask.resize(org_item.size, resample=Image.BILINEAR)
        # Apply mask
        logger.debug('Apply mask')
        empty = Image.new('RGBA', org_item.size)
        image = Image.composite(org_item, empty, mask)
        return image
