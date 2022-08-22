# -- coding: utf-8 --

from gpu_cls.utils.utils import indicators
import torch
import torch.utils.data
from tqdm import tqdm


def test_model(model, test_loader, Loss, epoch):
    model.eval()
    torch.cuda.empty_cache()
    test_loss = 0.0
    PA = 0
    FPR = 0
    TPR = 0
    tqrr = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_index, (data, target) in tqrr:
            # data = torch.unsqueeze(data, 1)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = Loss(output.squeeze(), target.squeeze())
            test_loss += loss.item()

            PA1, FPR1, TPR1 = indicators(output.squeeze(), target.squeeze()).get_PA_P_R()
            PA += PA1
            FPR += FPR1
            TPR += TPR1
            tqrr.set_description(
                " test Loss : {:.6f} \t PA :{:.6f} \t TNR :{:.6f} \t TPR :{:.6f}".format(loss.item(), PA1, FPR1, TPR1))

        PA /= len(test_loader)
        FPR /= len(test_loader)
        TPR /= len(test_loader)
        test_loss /= len(test_loader)

        print(" Epoch : {} \t test Loss : {:.6f} \t PA :{:.6f} \t TNR :{:.6f} \t TPR :{:.6f}".format(epoch,
                                                                                                 test_loss,
                                                                                                 PA,
                                                                                                 FPR, TPR))

        return test_loss, [PA, FPR, TPR]
