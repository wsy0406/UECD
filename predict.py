import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net


# can be changed according to config.txt
# exer_n = 17746
# knowledge_n = 123
# student_n = 4163
# exer_n = 714
# knowledge_n = 39
# student_n = 10000
# #########assist17
# exer_n = 3162
# knowledge_n = 102
# student_n = 1709
topk=20

def test():
    data_loader = ValTestDataLoader('test')
    device = torch.device('cpu')
    net = Net(student_n, exer_n, knowledge_n, topk, device)
    # net = Net(knowledge_n, exer_n, student_n)
    print('testing model...')
    for epoch in range(1,101):
    # epoch = 14
        data_loader.reset()
        load_snapshot(net, 'model/NCDM_UECD_epoch' + str(epoch))
        net = net.to(device)
        net.eval()

        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
            out_put = out_put.view(-1)
            # compute accuracy
            for i in range(len(labels)):
                if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += out_put.tolist()
            label_all += labels.tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / exer_count
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)
        print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
        with open('result/NCDM_UECD_test.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()



if __name__ == '__main__':


    # global student_n, exer_n, knowledge_n
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test()