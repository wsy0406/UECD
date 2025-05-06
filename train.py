
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import random
import os
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)
import pandas as pd
import csv
import time


# can be changed according to config.txt
# exer_n = 17746
# knowledge_n = 123
# student_n = 4163
# exer_n = 714
# knowledge_n = 39
# student_n = 10000
# can be changed according to command parameter
# device = torch.device(('cuda:3') if torch.cuda.is_available() else 'cpu')
epoch_n = 5
topk=20
batch_size=256




def set_seed(seed: int) -> int:
    r"""
    Set the seed for the random number generators.

    Parameters:
    -----------
    seed : int
        The seed to set. If seed is -1, a random seed between 0 and 2048 will be generated.

    Returns:
    --------
    seed : int
        The actual seed used.
    """
    if seed == -1:
        seed = random.randint(0, 2048)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Seed] >>> Set seed: {seed}")
    return seed


def train(device):
    data_loader = TrainDataLoader(topk,batch_size)
    net = Net(student_n, exer_n, knowledge_n,topk,device).to(device)
    # net = Net(knowledge_n, exer_n, student_n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_train_time = 0.0
    epoch_timings = []

    log_dir = './time_logs'

    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, f"training_time_NCDM_UECD_negloss_EAC.csv")

    # 初始化CSV文件
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'epoch',
            'start_time',
            'end_time',
            'duration(s)',
            'avg_duration(s)',
            'total_time(m)',
            'batch_count'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print('training model...')
    # loss_function = nn.NLLLoss()
    negloss_function = nn.MSELoss()
    loss_function = nn.BCELoss()
    final_loss = []
    final_neg_losses = []
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(epoch_n):

        epoch_start = time.perf_counter()

        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        neg_losses = 0.0


        if 'cuda' in str(device):
            torch.cuda.synchronize(device)

        while not data_loader.is_end():
            batch_count += 1
            all_neg_loss=0

            stu_ids, pos_ids, knowledge_embs, labels, neg_ids = data_loader.next_batch()
            stu_ids, knowledge_embs, labels = stu_ids.to(device),  knowledge_embs.to(device), labels.to(device)
            # print("labels",labels.size())             [256]
            can_pos_ids, can_neg_ids = net.item_sim_sample(pos_ids, neg_ids, topk, device)
            output_pos = net.forward(stu_ids, can_pos_ids, knowledge_embs)
            pos_loss = loss_function(output_pos.squeeze(), labels.float())

            # bpr_losses, qdcc_losses = net.forward(stu_ids, can_neg_ids, knowledge_embs,
            #                                                                can_pos_ids,
            #                                                                labels)  # [256, topk, 1] & [256,topk]

            output_negs, neg_scores, bpr_losses, qdcc_losses = net.forward(stu_ids, can_neg_ids, knowledge_embs, can_pos_ids, labels)

            # bpr_losses1, bpr_losses2 = net.forward(stu_ids, neg_ids,knowledge_embs, exer_knowledge_data, mis_neg_ids, pos_ids, labels)
            # bpr_losses = bpr_losses1 + bpr_losses2

            for i in range(output_negs.size(1)):
                each_neg=output_negs[torch.arange(output_negs.size(0)), i].view(-1,1)
                neg_score=neg_scores[torch.arange(neg_scores.size(0)), i].squeeze()

                neg_loss = negloss_function(each_neg.squeeze(), neg_score)
                all_neg_loss+=neg_loss

            loss = pos_loss + all_neg_loss/output_negs.size(1) + 0.1 * bpr_losses + qdcc_losses
            # loss = pos_loss + 0.1 * bpr_losses + qdcc_losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += pos_loss.item()
            # neg_losses += final_neg_loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss))
                running_loss = 0.0



        if 'cuda' in str(device):
            torch.cuda.synchronize(device)
        epoch_duration = time.perf_counter() - epoch_start


        total_train_time += epoch_duration
        epoch_timings.append(epoch_duration)


        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(epoch_start)),
                'end_time': time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(epoch_start + epoch_duration)),
                'duration(s)': round(epoch_duration, 2),
                'avg_duration(s)': round(np.mean(epoch_timings), 2),
                'total_time(m)': round(total_train_time / 60, 2),
                'batch_count': batch_count
            })

        time_msg = "[Epoch %d] Time: %.2fs | Avg: %.2fs | Total: %.1fmin" % (
            epoch,
            epoch_duration,
            np.mean(epoch_timings),
            total_train_time / 60
        )
        # print(time_msg)

        acc, rmse, auc = validate(net, epoch,device)
        save_snapshot(net, 'model/NCDM_UECD_epoch' + str(epoch + 1))

    final_time_msg = "\nTraining Time Summary:\nTotal: %.2f minutes\nMean per epoch: %.2f±%.2f seconds" % (
        total_train_time / 60,
        np.mean(epoch_timings),
        np.std(epoch_timings)
    )
    print(final_time_msg)

def validate(model, epoch, device):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n,topk,device)
    # net = Net(knowledge_n, exer_n, student_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        # input_stu_ids, input_exer_ids, input_knowledge_embs, labels,exer_knowledge_data = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        # output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,exer_knowledge_data)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('./result/NCDM_UECD_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return accuracy, rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])
    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    set_seed(2024)

    train(device)