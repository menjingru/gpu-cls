from gpu_cls.global_.global_path import *
from gpu_cls.global_.global_parameters import *
from gpu_cls.run.run_test import test_model
from gpu_cls.dataset.dataset import myDataset
import torch.utils.data
import pandas as pd
import time


def test(Loss, model=best_model_path, save=True):
    # dataloader
    dataset_test = myDataset(pre_npy, lei='ts')
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                              num_workers=num_workers)
    # test
    test_start = time.time()
    model = torch.load(model)
    test_loss, test_indicators = test_model(model, test_loader, Loss, EPOCH)

    # recoder
    if save:
        test_loss_list = [test_loss]
        test_loss_pd = pd.DataFrame(test_loss_list)
        test_loss_pd.to_excel(indicators_path + "/测试损失.xls")

        test_indicators_list = [test_indicators]
        test_indicators_pd = pd.DataFrame(test_indicators_list)
        test_indicators_pd.to_excel(indicators_path + "/测试验证指标[PA, TNR, TPR].xls")

        test_end = time.time()
        test_time = test_end - test_start
        print('Running time: %s Seconds' % test_time)
        test_time_list = [test_time]
        test_time_pd = pd.DataFrame(test_time_list)
        test_time_pd.to_excel(indicators_path + "/测试时间.xls")
