import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from data.transforms import *
import os

DETECTION_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCDetection(torch.utils.data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 root,
                 img_size=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'
                 ):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # basic augmentation(SSDAugmentation or BaseTransform)
        if self.transform is not None:
            # check labels
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 img_size=None,
                 image_sets=[('2012', 'train')],
                 transform=None,
                 target_transform=VOCAnnotationTransform()
                 ):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform


        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for year, name in image_sets:
            rootpath = os.path.join(os.getcwd(), self.root, 'VOC' + year)
            with open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt'), 'r') as f:
                for line in f:
                    self.ids.append((rootpath, line.strip()))       
        self.target_transform.set_class(DETECTION_CLASSES)
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # basic augmentation(SSDAugmentation or BaseTransform)
        if self.transform is not None:
            # check labels
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target


if __name__ == "__main__":
    from transforms import Augmentation, BaseTransform

    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    data_root = '/home/oem/MachineLearning/detect/datasets/VOCdevkit'
    transform = Augmentation(img_size, pixel_mean, pixel_std)
    transform = BaseTransform(img_size, pixel_mean, pixel_std)

    # dataset
    dataset = VOCDetection(
        root=data_root,
        img_size=img_size,
        image_sets=[('2007', 'trainval')],
        transform=transform
    )

    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)

        # to numpy
        image = im.permute(1, 2, 0).numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = (image * pixel_std + pixel_mean) * 255
        # to
        image = image.astype(np.uint8).copy()

        # draw bbox
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            image = cv2.rectangle(image, (int(xmin), int(
                ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.imshow('gt', image)
        cv2.waitKey(0)
