import torch
import random
import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import hdf5storage as hdf5
import cv2


BATCH_SIZE_TRAIN = 32
train_ratio = 0.01
patch_size = 13
pca_components = 32  # IP:32 PU:19 HU:32 HanChuan:32 Trento:32


# 读取数据
def loadData(datasetName):
    current_path = os.getcwd()
    dataSet_path = current_path + '/datasets'
    DATA_path = os.path.join(dataSet_path, datasetName)
    if datasetName == 'IP':
        data = sio.loadmat(os.path.join(DATA_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = sio.loadmat(os.path.join(DATA_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif datasetName == 'PU':
        data = sio.loadmat(os.path.join(DATA_path, 'PaviaU.mat'))['paviaU']
        label = sio.loadmat(os.path.join(DATA_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif datasetName == 'SA':
        data = sio.loadmat(os.path.join(DATA_path, 'Salinas_corrected.mat'))['salinas_corrected']
        label = sio.loadmat(os.path.join(DATA_path, 'Salinas_gt.mat'))['salinas_gt']
    elif datasetName == 'HU':
        data = sio.loadmat(os.path.join(DATA_path, 'Houston2013.mat'))['Houston']
        label = sio.loadmat(os.path.join(DATA_path, 'Houston2013_gt.mat'))['Houston_gt']
    elif datasetName == 'Trento':
        data = sio.loadmat(os.path.join(DATA_path, 'Trento.mat'))['Trento']
        label = sio.loadmat(os.path.join(DATA_path, 'Trento_gt.mat'))['Trento_gt']
    elif datasetName == 'Loukia':
        data = sio.loadmat(os.path.join(DATA_path, 'Loukia.mat'))['Loukia']
        label = sio.loadmat(os.path.join(DATA_path, 'Loukia_GT.mat'))['Loukia_GT']
    elif datasetName == 'HanChuan':
        data = sio.loadmat(os.path.join(DATA_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        label = sio.loadmat(os.path.join(DATA_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
    elif datasetName == 'KSC':
        data = sio.loadmat(os.path.join(DATA_path, 'KSC.mat'))['KSC']
        label = sio.loadmat(os.path.join(DATA_path, 'KSC_gt.mat'))['KSC_gt']
    elif datasetName == 'LongKou':
        data = sio.loadmat(os.path.join(DATA_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        label = sio.loadmat(os.path.join(DATA_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    
    return np.array(data, dtype=np.float32), np.array(label, dtype=np.int32)


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 可替换掉上面的 padWithZeros 函数
def zeroPadding_3D(X, margin, pad_depth=0):
    newX = np.pad(X, ((margin, margin), (margin, margin), (pad_depth, pad_depth)), 'constant', constant_values=0)
    # newX = np.pad(X, ((margin, margin), (margin, margin), (pad_depth, pad_depth)), 'reflect')
    # newX = np.pad(X, ((margin, margin), (margin, margin), (pad_depth, pad_depth)), 'symmetric')
    # newX = np.pad(X, ((margin, margin), (margin, margin), (pad_depth, pad_depth)), 'wrap')

    return newX


# 在每个像素周围提取 patch
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = zeroPadding_3D(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


# 指定采样某些类确定数量的训练集和测试集
def train_test_sample_fix(X, y, train_ratio):
    train_indices = []
    test_indices = []
    for c in np.unique(y):
        indices = np.nonzero(y == c)
        class_indices = list(*indices)
        real_train_ratio = train_ratio
        if real_train_ratio > len(class_indices):
            real_train_ratio = 5
        # if c == 0 or c == 6 or c == 8:
        #     real_train_ratio = 100
        train_indices += random.sample(class_indices, real_train_ratio)
        test_indices += list(set(class_indices) - set(train_indices))
        
    index_train = np.array(train_indices)
    index_test = np.array(test_indices)
    np.random.shuffle(index_train)
    np.random.shuffle(index_test)
    X_train = X[index_train]
    y_train = y[index_train]
    X_test = X[index_test]
    y_test = y[index_test]
    return X_train, X_test, y_train, y_test


# 划分训练集和测试集
def splitTrainTestSet(X, y, train_ratio, randomState=2024):
    # 按比例划分
    if train_ratio > 0 and train_ratio <= 1:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            train_size=train_ratio,
                                                            random_state=randomState,
                                                            stratify=y)
    # 按数量划分
    elif train_ratio > 1:
        X_train, X_test, y_train, y_test = train_test_sample_fix(X, y, train_ratio)
    else:
        raise ValueError('train_ratio must > 0')

    return X_train, X_test, y_train, y_test


# 随机概率数据增强
def hsi_augment(data):
    do_augment = np.random.random()
    if do_augment > 0.5:
        prob = np.random.random()
        if 0 <= prob <= 0.2:
            data = np.fliplr(data)
        elif 0.2 < prob <= 0.4:
            data = np.flipud(data)
        elif 0.4 < prob <= 0.6:
            data = np.rot90(data, k=1)
        elif 0.6 < prob <= 0.8:
            data = np.rot90(data, k=2)
        elif 0.8 < prob <= 1.0:
            data = np.rot90(data, k=3)
    return data


class CenterResizeCrop(object):
    def __init__(self, scale_begin=17, windowsize=27):
        self.scale_begin = scale_begin
        self.windowsize = windowsize
    
    def __call__(self, image):
        length = np.array(range(self.scale_begin, self.windowsize+1, 2))  # 从17到27,每隔1步取一个数
        row_center = int((self.windowsize-1)/2)  # 中心位置
        col_center = int((self.windowsize-1)/2)   
        row = image.shape[1]
        col = image.shape[2]
        s = np.random.choice(length, size=1)  # 随机选择一个数
        halfsize_row = int((s-1)/2)
        halfsize_col = int((s-1)/2)             
        r_image = image[:, row_center-halfsize_row : row_center+halfsize_row+1, col_center-halfsize_col : col_center+halfsize_col+1]  # 从中心裁剪(大patch里面挖小patch)
        r_image = np.transpose(cv2.resize(np.transpose(r_image, [1,2,0]), (row, col)), [2,0,1])  # 将裁剪的小patch放大到原始大小(采用默认的双线性插值)
        return r_image


class HyperData(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, label, transfor):
        self.dataset = dataset.astype(np.float32)
        self.transfor = transfor
        self.labels = []
        for n in label:
            self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.transfor(self.dataset[index, :, :, :]))).to(torch.float32)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.labels)



def create_data_loader(datasetName, result_path):
    X, gt = loadData(datasetName)

    X_pca = applyPCA(X, numComponents=pca_components)
    # ----------------------------------------------------------------------------------
    # --------------------------------- 归一化 ------------------------------------------
    # ----------------------------------------------------------------------------------
    # X = np.asarray(X, dtype=np.float32)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    # mean_by_c = np.mean(X, axis=(0, 1))
    # for c in range(X.shape[-1]):
    #     X[:, :, c] = X[:, :, c] - mean_by_c[c]
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # X_pca = X

    Height, Width, PCAComponents = X_pca.shape
    X_pca, y_all = createImageCubes(X_pca, gt, windowSize=patch_size)
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, train_ratio)
    Xval, Xtest, yval, ytest = splitTrainTestSet(Xtest, ytest, 0.1)

    X = X_pca.reshape(-1, patch_size, patch_size, PCAComponents)  # 确定形状
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, PCAComponents)  # 确定形状
    Xval = Xval.reshape(-1, patch_size, patch_size, PCAComponents)  # 确定形状
    Xtest = Xtest.reshape(-1, patch_size, patch_size, PCAComponents)  # 确定形状

    X = torch.from_numpy(X.transpose(0, 3, 1, 2)).to(torch.float32)
    # Xtrain = torch.from_numpy(Xtrain.transpose(0, 3, 1, 2)).to(torch.float32)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xval = torch.from_numpy(Xval.transpose(0, 3, 1, 2)).to(torch.float32)
    Xtest = torch.from_numpy(Xtest.transpose(0, 3, 1, 2)).to(torch.float32)

    # ----------------------------------------------------------------------------------
    # ----------------------------- 训练集做数据增强 ------------------------------------
    # ----------------------------------------------------------------------------------
    # for i in range(Xtrain.shape[0]):
    #     Xtrain[i] = hsi_augment(Xtrain[i])
    # Xtrain = torch.from_numpy(Xtrain.transpose(0, 3, 1, 2)).to(torch.float32)
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # 统计每类训练、测试样本数
    class_num = (np.max(ytrain) + 1).astype(int)
    each_class_num_train = dict()
    each_class_num_test = dict()
    each_class_num_val = dict()
    for i in range(class_num):
        each_class_num_train[i+1] = np.sum(ytrain == i)
        each_class_num_val[i+1] = np.sum(yval == i)
        each_class_num_test[i+1] = np.sum(ytest == i)
    print('训练集每类样本数：', each_class_num_train)
    print('验证集每类样本数：', each_class_num_val)
    print('测试集每类样本数：', each_class_num_test)
    
    with open(result_path + '/EachClassNum_Train-Test.txt', 'a', encoding='utf-8') as f:
        f.write('训练集每类的样本数: ' + str(each_class_num_train) + '\n\n')
        f.write('验证集每类的样本数: ' + str(each_class_num_val) + '\n\n')
        f.write('测试集每类的样本数: ' + str(each_class_num_test) + '\n' + '--'*50 + '\n')

    X = DS(X, y_all)
    # trainset = DS(Xtrain, ytrain)
    transform_train = CenterResizeCrop(scale_begin=5, windowsize=patch_size)
    trainset = HyperData(Xtrain, ytrain, transform_train)
    
    valset = DS(Xval, yval)
    testset = DS(Xtest, ytest)

    all_loader = torch.utils.data.DataLoader(dataset=X,
                                             batch_size=BATCH_SIZE_TRAIN,
                                             shuffle=False,
                                             num_workers=0
                                             )
    
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0
                                               )
    validation_loader = torch.utils.data.DataLoader(dataset=valset,
                                                     batch_size=BATCH_SIZE_TRAIN,
                                                     shuffle=False,
                                                     num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0
                                               )
    return train_loader, validation_loader, test_loader, all_loader, gt, PCAComponents


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.data = torch.Tensor(X)
        self.label = torch.LongTensor(y)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.len

