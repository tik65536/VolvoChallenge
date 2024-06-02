import os
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import importlib
import importlib.util
import yaml
import numpy as np
import pandas as pd


class TrainTestDataSet():

    def __init__(self, dataset_config=None):
        with open(dataset_config, 'r') as file:
            dataset_config = yaml.safe_load(file)
        dataconfig = dataset_config['data']

        self.dataloadername = dataconfig['loader']
        self.filetype = dataconfig['filetype']
        self.Dataloader = getattr(importlib.import_module('utils.dataloader'), self.dataloadername)
        self.trainDataPath = None
        self.testDataPath = None
        self.valDataPath = None
        self.trainset = None
        self.testset = None
        self.valset = None
        traindatasetpath = dataconfig['training']['data']
        testdatasetpath = dataconfig['testing']['data']
        if (os.path.isdir(traindatasetpath)):
            traindatasetpath = [f'{traindatasetpath}/{file}' for file in os.listdir(traindatasetpath) if (file.split('.')[-1] == self.filetype)]
            self.trainset = self.Dataloader(traindatasetpath)
            self.trainDataPath = traindatasetpath
        if (os.path.isdir(testdatasetpath)):
            testdatasetpath = [f'{testdatasetpath}/{file}' for file in os.listdir(testdatasetpath) if (file.split('.')[-1] == self.filetype)]
            self.testset = self.Dataloader(testdatasetpath)
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
        if (self.tranformation is not None):
            self.module =  self.tranformation['module']
            for key in self.tranformation.keys():
                if (key != 'module'):
                    t = getattr(importlib.import_module(self.module), key)
                    t = t(**self.tranformation[key]) if ('torchvision' in self.module.lower()) else {'fun':t, 'param':self.tranformation[key]}
                    self.tranforms.append(t)
            if ('torchvision' in self.module.lower()):
                self.tranforms = Compose(self.tranforms)

    def __len__(self):
        return len(self.datasetpath)

    def __getitem__(self, index):
        filename = self.datasetpath[index].split('/')[-1]
        with rasterio.open(f'{self.datasetpath[index]}') as ds:
            dataimg = ds.read()
        with rasterio.open(f'{self.datasetlabelpath}/{filename}') as ds:
            labelimg = ds.read()
        if (self.tranformation is not None):
            if ('torchvision' in self.module.lower()):
                dataimg = torch.from_numpy(dataimg)
                labelimg = torch.from_numpy(labelimg)
                dataimg = self.tranforms(dataimg)
                labelimg = self.tranforms(labelimg) if (self.apply) else labelimg
            else:
                for transform in self.tranforms:
                    dataimg = transform['fun'](dataimg,**transform['param'])
                    labelimg = transform['fun'](labelimg,**transform['param'])
                dataimg = torch.from_numpy(dataimg)
                labelimg = torch.from_numpy(labelimg)
            return (dataimg, labelimg, filename)
        else:
            dataimg = torch.from_numpy(dataimg)
            labelimg = torch.from_numpy(np.float32(labelimg))
            return (dataimg, labelimg, filename)

class SKImgDataLoader(Dataset):

    def __init__(self, datasetpath, datasetlabelpath, testsetApplyTransformation=True, tranformations=None,):
        super().__init__()
        print('Fouce to use skimage for resizing to (1,512,512)')
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

class VolvoCSVDataLoader(Dataset):

    def __init__(self, datasetpath):
        super().__init__()
        # assuming label is contained in the csv file.
        self.datasetpath = datasetpath
        self.dataframe = [ pd.read_csv(f) for f in self.datasetpath ]
        self.dataframe = pd.concat(self.dataframe, axis=0, ignore_index=True)
        nan_cols=self.dataframe.columns.values[np.where(np.any(self.dataframe.isna(),axis=0)==True)]
        self.dataframe = self.dataframe.loc[:,~self.dataframe.columns.isin(nan_cols)]
        chassisid = len(self.dataframe.groupby("ChassisId_encoded").groups.keys())
        maxtimestep = np.max(self.dataframe.groupby("ChassisId_encoded")["ChassisId_encoded"].count())
        readings_cols =  self.dataframe.columns[5:]
        self.dataframe.risk_level = self.dataframe.risk_level.astype('category').cat.codes
        print(f'Num of ChassisID {chassisid}, Max timestep {maxtimestep}, Feature len {len(readings_cols)}')
        self.data = np.zeros((chassisid,maxtimestep,len(readings_cols)),dtype='float32')
        self.label = []
        self.grp_idx = []
        for i, grp in enumerate(self.dataframe.groupby('ChassisId_encoded')):
            self.grp_idx.append(grp[0])
            tmp = grp[1][readings_cols]
            self.label += grp[1].iloc[:,4].values.tolist()
            self.data[i,:len(tmp),:]=tmp

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (self.data[index,:,:],self.label[index],'')

