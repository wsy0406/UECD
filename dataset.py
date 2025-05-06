

import os, json, ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import Net


class MyDataset(Dataset):

    def __init__(
        self,
        path:str = "data/Eedi",
        filename = "neg_exer_ours.json",
        config_file = 'config_Eedi.txt',
        topk: int = 20
    ):
        super().__init__()

        data_file = os.path.join(
            path, filename
        )
        config_file = config_file

        with open(data_file, encoding='utf8') as i_f:
            # {
            #     'user_id': 1615,
            #     'exer_id': 12977,
            #     'score': 1.,
            #     'knowledge_code': [83]
            #     'neg_exer_id': [1023, 1024, 1025]
            # }
            self.data = json.load(i_f)

        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n) # number of knowledge points

        self.topk = topk

        self.k_difficulty = None
        self.e_discrimination = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        piece = self.data[idx]
        knowledge_emb = [0.] * self.knowledge_dim

        if isinstance(piece['knowledge_code'], float):
            piece['knowledge_code'] = [int(piece['knowledge_code'])]
        if isinstance(piece['knowledge_code'], str):
            piece['knowledge_code'] = ast.literal_eval(piece['knowledge_code'])
        piece['knowledge_code'] = [int(k) for k in piece['knowledge_code']]
        for knowledge_code in piece['knowledge_code']:
            knowledge_emb[knowledge_code - 1] = 1.0
        y = int(piece['score'])
        stu_id = int(piece['user_id'])
        exer_id = int(piece['exer_id'])
        neg_pos_exer_ids = [int(id_) for id_ in piece['neg_exer_ids']]

        neg_pos_exer_ids = self.item_sim_sample(exer_id, neg_pos_exer_ids, self.topk, 'cpu')

        return stu_id, exer_id, torch.LongTensor(knowledge_emb), int(y), torch.LongTensor(neg_pos_exer_ids)

    def update(
        self,
        k_difficulty: nn.Embedding,
        e_discrimination: nn.Embedding
    ):
        self.k_difficulty = k_difficulty.weight.data.clone().cpu()
        self.e_discrimination = e_discrimination.weight.data.clone().cpu()

    def item_sim_sample(self, pos_id, neg_ids, topk, device):
        _, can_item = self.item_similarity(pos_id, neg_ids, topk, device)
        return can_item.squeeze().tolist()

    def item_similarity(self,pos_id, neg_id,topk, device):
        # preparation: params and id tensor

        device = 'cpu'
        inter_item_tensor = torch.tensor([pos_id])
        non_item_tensor = torch.tensor(neg_id).unsqueeze(-1)

        # difficulty
        inter_beta_e = self.k_difficulty[inter_item_tensor]
        non_beta_e = self.k_difficulty[non_item_tensor]
        dif_score = torch.sum(inter_beta_e * non_beta_e, dim=[1, 2])

        # discrimination
        inter_alp_e = self.e_discrimination[inter_item_tensor]
        non_alp_e = self.e_discrimination[non_item_tensor]
        dis_score = torch.sum(inter_alp_e * non_alp_e, dim=[1, 2])

        # sum of two embedding score
        sim_score = (F.softmax(dif_score, dim=-1) + F.softmax(dis_score, dim=-1)) / 2
        sampled_index = torch.multinomial(sim_score, topk, replacement=True).to(device)
        # if label == 1:
        #     sample_label = sim_score[sampled_index]
        # else:
        #     sample_label = 1 - sim_score[sampled_index]


        # return inter_item_tensor, non_item_tensor[sampled_index], sample_label
        return inter_item_tensor, non_item_tensor[sampled_index]

