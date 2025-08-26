import torch
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
import matplotlib.colors as mcolors
from operator import truediv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Data_Loader import create_data_loader
from self_similarity import SSM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 固定随机数
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False  # 为了复现
    torch.backends.cudnn.deterministic = True  # 为了复现


# 保存图片
def saveFig(img, imgname):
    colors = [    
    (0.0, 0.0, 0.0),  # 黑色0
    (0.0, 1.0, 0.0),  # 绿色1
    (0.0, 0.0, 1.0),  # 蓝色2
    (1.0, 1.0, 0.0),  # 黄色3
    (1.0, 0.0, 1.0),  # 紫色4
    (0.0, 1.0, 1.0),  # 青色5
    (1.0, 0.5, 0.0),  # 橙色6
    (0.5, 0.5, 0.5),  # 中灰色7
    (1.0, 1.0, 1.0),  # 白色8
    (0.5, 0.5, 0.0),  # 橄榄色9
    (0.5, 0.0, 0.5),  # 紫红色10
    (1.0, 0.5, 0.5),  # 粉红色11
    (0.5, 1.0, 0.5),  # 浅绿色12
    (0.5, 0.5, 1.0),  # 浅蓝色13
    (1.0, 0.75, 0.0), # 金色14
    (0.75, 0.0, 1.0), # 深紫色15
    (1.0, 0.0, 0.0)  # 红色16
    ]
    # 创建颜色映射
    cmap = mcolors.ListedColormap(colors)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(imgname, dpi=600, pad_inches=0.0, bbox_inches='tight')
    plt.close()


# 保存测试结果
def log(OA, AA, KAPPA, Each_Time_ACC, TRAIN_TIME, TEST_TIME, filename):
    average_OA = round(np.mean(OA), 2)
    average_AA = round(np.mean(AA), 2)
    average_KAPPA = round(np.mean(KAPPA), 2)
    average_Each_Time_ACC = [round(each, 2) for each in np.mean(Each_Time_ACC, axis=0)]
    average_train_time = round(np.mean(TRAIN_TIME), 2)
    average_test_time = round(np.mean(TEST_TIME), 2)

    std_OA = round(np.std(OA), 2)
    std_AA = round(np.std(AA), 2)
    std_KAPPA = round(np.std(KAPPA), 2)
    std_Each_Time_ACC = [round(each, 2) for each in np.std(Each_Time_ACC, axis=0)]
    std_train_time = round(np.std(TRAIN_TIME), 2)
    std_test_time = round(np.std(TEST_TIME), 2)

    with open(filename, 'a') as f:
        f.write('OA: ' + str(OA) + '\n')
        f.write('AA: ' + str(AA) + '\n')
        f.write('KAPPA: ' + str(KAPPA) + '\n')
        f.write('Each_Time_ACC:\n' + str(Each_Time_ACC) + '\n')
        f.write('Train_time: ' + str(TRAIN_TIME) + '\n')
        f.write('Test_time: ' + str(TEST_TIME) + '\n')
        f.write('---'*10 + '\n')

        f.write('Final_OA:  ' + str(average_OA) + ' +- ' + str(std_OA) + '\n')
        f.write('Final_AA:  ' + str(average_AA) + ' +- ' + str(std_AA) + '\n')
        f.write('Final_KAPPA:  ' + str(average_KAPPA) + ' +- ' + str(std_KAPPA) + '\n')
        f.write('Final_Each_Time_ACC:\n' + str(average_Each_Time_ACC) + '\n +- \n' + str(std_Each_Time_ACC) + '\n')
        f.write('\nFinal_Train_time:  ' + str(average_train_time) + ' +- ' + str(std_train_time) + '\n')
        f.write('Final_Test_time:  ' + str(average_test_time) + ' +- ' + str(std_test_time) + '\n')
        f.write('************************************ OVER! *************************************' + '\n')


# 画预测图
def Draw_pred_Picture(all_loader, gt, model, result_path):
    pred = np.array([])
    pred_pic = np.zeros(gt.shape)
    for step, (batch_all_data, batch_all_y) in enumerate(all_loader):
        batch_all_data = batch_all_data.cuda()
        batch_all_y = batch_all_y.cuda()
        batch_out = model(batch_all_data)

        _, batch_pred = torch.max(batch_out, dim=1)
        pred = np.append(pred, batch_pred.detach().cpu().numpy())

    pred = pred.reshape(-1)
    pred = pred + 1
    pred_pic[gt!=0] = pred
    
    saveFig(pred_pic, result_path + '/pred.png')
    saveFig(gt, result_path + '/target.png')


def AA_and_EachClassAccuracy(confusion):
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


# 验证
def validation(validation_loader, model):
    model.eval()
    for step, (batch_test_data, batch_test_y) in enumerate(validation_loader):
        batch_test_data = batch_test_data.cuda()
        batch_test_y = batch_test_y.cuda()
        batch_out = model(batch_test_data)

        _, batch_pred = torch.max(batch_out, dim=1)
        if step == 0:
            pred = batch_pred
            target = batch_test_y
        else:
            pred = torch.cat((pred, batch_pred), dim=0)
            target = torch.cat((target, batch_test_y), dim=0)
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(-1)
    target = target.detach().cpu().numpy()
    target = target.reshape(-1)

    oa = accuracy_score(target, pred)

    return oa*100#round(oa*100, 2)


# 测试
def test(test_loader, model):
    Test_time = 0
    for step, (batch_test_data, batch_test_y) in enumerate(test_loader):
        batch_test_data = batch_test_data.cuda()
        batch_test_y = batch_test_y.cuda()
        start_test = time.time()
        batch_out = model(batch_test_data)
        end_test = time.time()
        Test_time += end_test - start_test

        _, batch_pred = torch.max(batch_out, dim=1)
        if step == 0:
            pred = batch_pred
            target = batch_test_y
        else:
            pred = torch.cat((pred, batch_pred), dim=0)
            target = torch.cat((target, batch_test_y), dim=0)
    # --------------------------------------------------------------------------------------------
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(-1)
    target = target.detach().cpu().numpy()
    target = target.reshape(-1)

    confusion = confusion_matrix(target, pred)
    oa = accuracy_score(target, pred)
    each_acc, aa = AA_and_EachClassAccuracy(confusion)
    kappa = cohen_kappa_score(target, pred)

    return confusion, round(oa*100, 2), [round(each, 2) for each in each_acc*100], round(aa*100, 2), round(kappa*100, 2), round(Test_time, 2)


# 训练
def train(datasetName, EPOCH, runTimes, optimizer_weight_decay, optimizer_LR):
    # set_seed(1024)
    unfoldSize=11

    method_Name = '__my_method'
    current_path = os.getcwd()
    result_path = current_path + '/' + method_Name + '/results/' + datasetName
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # ------------------ 训练集、测试集 loader ---------------------
    train_loader, validation_loader, test_loader, all_loader, gt, PCAComponents = create_data_loader(datasetName, result_path)
    Classes = np.max(gt)
    # ------------------------ 初始化 --------------------------
    KAPPA = []
    OA = []
    AA = []
    Each_Time_ACC = np.zeros((runTimes, Classes))
    TRAIN_TIME = []
    TEST_TIME = []
    model_path = result_path + '/net_params.pkl'
    train_acc = np.zeros(EPOCH)
    val_acc = np.zeros(EPOCH)
    model = SSM(in_ch=PCAComponents, mid_ch=32, unfold_size=unfoldSize, ksize=unfoldSize, nc=Classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_LR, weight_decay=optimizer_weight_decay)
    loss_func = nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  # 每过step_size个epoch，LR就乘以gamma
    
    torch.cuda.synchronize()  # 同步CPU和GPU之间的计算  # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    for run in range(runTimes):
        val_time = 0
        best_val = 0
        start_train = time.time()
        # ---------------------------- 训练 --------------------------------
        for epoch in tqdm(range(EPOCH)):
            model.train()
            for step, (batch_train_data, batch_train_y) in enumerate(train_loader):
                batch_train_data = batch_train_data.cuda()
                batch_train_y = batch_train_y.cuda()
                batch_train_out = model(batch_train_data)  # 输入shape为(b,c,h,w) (64,30,13,13)  输出torch.Size([64, 16])
                loss = loss_func(batch_train_out, batch_train_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            # ----------------------------- 比较精度曲线 --------------------------------
            start_val = time.time()
            train_acc[epoch] = validation(train_loader, model)
            val_acc[epoch] = validation(validation_loader, model)
            if val_acc[epoch] > best_val:
                best_val = val_acc[epoch]
                torch.save(model.state_dict(), model_path)
            end_val = time.time()
            val_time += end_val - start_val

        torch.cuda.synchronize()
        end_train = time.time()
        Train_time = end_train - start_train - val_time
        print('Train_time: ', round(Train_time, 2))
        
        # 绘制 训练集和验证集 准确率曲线
        plt.plot(train_acc, label='Train')
        plt.plot(val_acc, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(result_path + '/Accuracy_Curve_{}.png'.format(run+1))
        plt.close()

        model.load_state_dict(torch.load(model_path))
        # ----------------------------- 测试 --------------------------------
        model.eval()
        with torch.no_grad():
            confusion, oa, each_acc, aa, kappa, Test_time = test(test_loader, model)
            print('Test_time: ', Test_time)
            Draw_pred_Picture(all_loader, gt, model, result_path)
        OA.append(oa)
        AA.append(aa)
        KAPPA.append(kappa)
        Each_Time_ACC[run, :] = each_acc
        TRAIN_TIME.append(Train_time)
        TEST_TIME.append(Test_time)

        print("OA = ", oa)
        print("AA = ", aa)
        print("KAPPA = ", kappa)
        print("Each_ACC = ", each_acc)
    log(OA, AA, KAPPA, Each_Time_ACC, TRAIN_TIME, TEST_TIME, filename=result_path + '/result.txt')


if __name__ == '__main__':
    datasetNames = ['IP', 'PU', 'HanChuan', 'Trento', 'SA', 'HU', 'LongKou']  # HSI数据集
    datasetName = datasetNames[0]

    EPOCH = 100
    runTimes = 5
    optimizer_weight_decay = 5e-3
    optimizer_LR = 5e-4
    # torch.cuda.empty_cache()
    train(datasetName, EPOCH, runTimes, optimizer_weight_decay, optimizer_LR)