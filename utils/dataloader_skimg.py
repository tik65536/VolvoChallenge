import os
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
from skimage.transform import resize
import importlib
from operator import itemgetter
import importlib.util
import yaml
import numpy as np


class TrainTestDataSet():

    def __init__(self, dataset_config=None):
        with open(dataset_config, 'r') as file:
            dataset_config = yaml.safe_load(file)
        dataconfig = dataset_config['data']
        print('Fouce to use skimage for resizing to (1,512,512) ')
        self.dataloadername = dataconfig['loader']
        self.filetype = dataconfig['filetype']
        self.tranformation = dataconfig['transforms']
        self.testsetApplyTransformation = dataconfig['testsettransforms']
        self.Dataloader = getattr(importlib.import_module('utils.dataloader'), self.dataloadername)
        self.trainDataPath = None
        self.testDataPath = None
        self.valDataPath = None
        self.trainset = None
        self.testset = None
        self.valset = None
        if (dataconfig.get('testing', '') == ''):
            # Perform Train Test split
            datasetpath = dataconfig['training']['data']
            labelpath = dataconfig['training']['label']
            trainratio = dataconfig.get('trainratio', 0.8)
            if ((os.path.isdir(datasetpath)) and (os.path.isdir(labelpath))):
                datasetpath = [f'{datasetpath}/{file}' for file in os.listdir(datasetpath) if (file.split('.')[-1] == self.filetype)]
                split = int(np.ceil(trainratio * len(datasetpath)))
                idx = np.arange(len(datasetpath),dtype=int)
                np.random.shuffle(idx)
                self.trainset = self.Dataloader(itemgetter(*idx[:split])(datasetpath), labelpath, self.testsetApplyTransformation, self.tranformation)
                self.testset = self.Dataloader(itemgetter(*idx[split:])(datasetpath), labelpath, self.testsetApplyTransformation, self.tranformation)
                self.trainDataPath = itemgetter(*idx[:split])(datasetpath)
                self.testDataPath = itemgetter(*idx[split:])(datasetpath)
        else:
            # load train test data set individually
            traindatasetpath = dataconfig['training']['data']
            trainlabelpath = dataconfig['training']['label']
            testdatasetpath = dataconfig['testing']['data']
            testlabelpath = dataconfig['testing']['label']
            if ((os.path.isdir(traindatasetpath)) and (os.path.isdir(trainlabelpath))):
                traindatasetpath = [f'{traindatasetpath}/{file}' for file in os.listdir(traindatasetpath) if (file.split('.')[-1] == self.filetype)]
                if (dataconfig.get('trainratio')):
                    trainratio = dataconfig.get('trainratio', 0.8)
                    split = int(np.ceil(trainratio * len(traindatasetpath)))
                    idx = np.arange(len(traindatasetpath),dtype=int)
                    np.random.shuffle(idx)
                    self.trainset = self.Dataloader(itemgetter(*idx[:split])(traindatasetpath), trainlabelpath, self.testsetApplyTransformation, self.tranformation)
                    self.valset = self.Dataloader(itemgetter(*idx[split:])(traindatasetpath), trainlabelpath, self.testsetApplyTransformation, self.tranformation)
                    self.trainDataPath = itemgetter(*idx[:split])(traindatasetpath)
                    self.valDataPath = itemgetter(*idx[split:])(traindatasetpath)
                else:
                    self.trainset = self.Dataloader(traindatasetpath, trainlabelpath, self.testsetApplyTransformation, self.tranformation)
                    self.trainDataPath = traindatasetpath
                    if (dataconfig.get('validation')):
                        valdatapath = dataconfig['validation']['data']
                        vallabelpath = dataconfig['validation']['label']
                        valdatapath = [f'{valdatapath}/{file}' for file in os.listdir(valdatapath) if (file.split('.')[-1] == self.filetype)]
                        self.valset = self.Dataloader(valdatapath, vallabelpath, self.testsetApplyTransformation, self.tranformation)
                        self.valDataPath = valdatapath
            if ((os.path.isdir(testdatasetpath)) and (os.path.isdir(testlabelpath))):
                testdatasetpath = [f'{testdatasetpath}/{file}' for file in os.listdir(testdatasetpath) if (file.split('.')[-1] == self.filetype)]
                self.testset = self.Dataloader(testdatasetpath, testlabelpath, self.testsetApplyTransformation, self.tranformation)
                self.testDataPath = testdatasetpath


class ImgDataLoader(Dataset):

    def __init__(self, datasetpath, datasetlabelpath, testsetApplyTransformation=True, tranformations=None,):
        super().__init__()

        self.datasetpath = datasetpath
        self.datasetlabelpath = datasetlabelpath
        self.tranformation = tranformations
        self.tranforms = []
        self.apply = testsetApplyTransformation
        self.module = None

    def __len__(self):
        return len(self.datasetpath)

    def __getitem__(self, index):
        filename = self.datasetpath[index].split('/')[-1]
        with rasterio.open(f'{self.datasetpath[index]}') as ds:
            dataimg = ds.read()
            dataimg = resize(dataimg,output_shape=(1,512,512),mode='constant', preserve_range=True)
            dataimg = torch.FloatTensor(dataimg)
        with rasterio.open(f'{self.datasetlabelpath}/{filename}') as ds:
            labelimg = ds.read()
            labelimg = resize(labelimg,output_shape=(1,512,512), preserve_range=True, order=0)
            labelimg = torch.FloatTensor(labelimg)
        #if ('torchvision' in self.module.lower()):
        #    dataimg = torch.from_numpy(dataimg)
        #    labelimg = torch.from_numpy(np.float32(labelimg))
        #   dataimg = self.tranforms(dataimg)
        #   labelimg = self.tranforms(labelimg) if (self.apply) else labelimg
        #else:
        #    for transform in self.tranforms:
        #        dataimg = transform['fun'](dataimg,**transform['param'])
        #        labelimg = transform['fun'](labelimg, **transform['param'])
        return (dataimg, labelimg, filename)


if __name__ == "__main__":
    d = ImgDataLoader(mode="Train")
    d.__getitem__(1)
