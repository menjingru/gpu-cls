from gpu_cls.global_.global_path import *
from gpu_cls.global_.global_parameters import *
from gpu_cls.run.run_train import train_model
from gpu_cls.run.run_valid import valid_model
from gpu_cls.dataset.dataset import myDataset
from gpu_cls.utils.utils import early_or_not
import torch.utils.data
import pandas as pd
import xlwt
import time


def train(model, optimizer, Loss, val_or_not=True, early_stop=True ):
    # dataloader
    dataset_train = myDataset(pre_npy, lei='tr')
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE, shuffle=True, #drop_last=True,
                                               num_workers=num_workers)


    dataset_valid = myDataset(pre_npy, lei='val')
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=BATCH_SIZE, shuffle=True, #drop_last=True,
                                               num_workers=num_workers)
    # train
    train_start = time.time()
    train_loss_list = []
    valid_loss_list = []
    minnum = 0

    for epoch in range(1, EPOCH + 1):
        train_loss = train_model(model, train_loader, optimizer, Loss, epoch)

        # recording
        train_loss_list.append(train_loss)
        train_loss_pd = pd.DataFrame(train_loss_list)
        train_loss_pd.to_excel(indicators_path + "/训练损失.xls")
        torch.save(model, model_path + '/train_model.pth')
        torch.cuda.empty_cache()
        # n training 1 validation
        n_tr_one_val = 1
        if val_or_not:
            if epoch % n_tr_one_val == 0:
                valid_loss, valid_indicators = valid_model(model=model, val_loader=valid_loader, Loss=Loss, epoch=epoch)
                print("valid_loss", valid_loss)
                valid_loss_list.append(valid_loss)
                valid_loss_pd = pd.DataFrame(valid_loss_list)
                valid_loss_pd.to_excel(indicators_path + "/验证损失.xls")
                if early_stop:
                    if early_or_not(valid_loss_list):
                        print('val loss does not decline in 10 epoch, early stop!')
                        break
                if epoch == n_tr_one_val:
                    torch.save(model, model_path + '/best_model.pth')
                    minnum = valid_loss
                    print("minnum", minnum)
                elif valid_loss < minnum:
                    print("valid_loss < minnum", valid_loss, "<", minnum)
                    minnum = valid_loss
                    torch.save(model, model_path + '/best_model.pth')
                    valid_indicators_pd = pd.DataFrame(valid_indicators)
                    valid_indicators_pd.to_excel(
                        indicators_path + "/目前为止最合适的model指标：第%d个epoch的验证指标[ PA, TNR, TPR].xls" % epoch)
                else:
                    pass
            torch.cuda.empty_cache()


    train_end = time.time()
    train_time = train_end - train_start
    print('Running time: %s Seconds' % train_time)
    time_list = [train_time]
    train_time_pd = pd.DataFrame(time_list)
    train_time_pd.to_excel(indicators_path + "/总epoch的训练时间.xls")
