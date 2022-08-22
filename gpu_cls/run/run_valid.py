# -- coding: utf-8 --

from gpu_cls.utils.utils import indicators
import torch
import torch.utils.data
from tqdm import tqdm


def valid_model(model, val_loader, Loss, epoch):
    """
    验证模型
    :param model: best model
    :param val_loader: 验证集
    :param Loss: loss function
    :param epoch:
    :return:
    """
    model.eval()
    test_loss = 0.0
    PA = 0
    FPR = 0
    TPR = 0
    tqrr = tqdm(enumerate(val_loader))
    with torch.no_grad():
        for batch_index, (data, target) in tqrr:
            # data = torch.unsqueeze(data, 1)
            data, target = data.cuda(), target.cuda()
            torch.cuda.empty_cache()
            output = model(data)
            loss = Loss(output.squeeze(), target.squeeze())
            test_loss += loss.item()

            PA1, FPR1, TPR1 = indicators(output.squeeze(), target.squeeze()).get_PA_P_R()
            PA += PA1
            FPR += FPR1
            TPR += TPR1

        PA /= len(val_loader)
        FPR /= len(val_loader)
        TPR /= len(val_loader)
        test_loss /= len(val_loader)

        print(
            " Epoch : {} \t valid Loss : {:.6f} \t PA :{:.6f} \t TNR :{:.6f} TPR: {:.6f}".format(epoch, test_loss, PA, FPR,
                                                                                               TPR))

        return test_loss, [PA, FPR, TPR]
