import torch
from torch.nn import Parameter
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import MATLoss
import numpy as np
import torch.nn.functional as F
import math
import pickle
from gat import GAT
import dgl

class DocREModel(nn.Module):

    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = MATLoss()
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size + 97, config.num_labels)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.feature_fusion_linear = nn.Linear(768 * 2, 768)
        self.feature_fusion_lstm = nn.LSTM(input_size=768 * 2, hidden_size=768, bidirectional=True, batch_first=True,
                                           num_layers=2, dropout=0.2)
        docred_adj = pickle.load(open('DocRED_adj.pkl', 'rb'))
        A = self.gen_dgl_graph(docred_adj)
        A = A.int().to(0)
        self.gat = GAT(g=A,
                       num_layers=2,
                       in_dim=768,
                       num_hidden=500,
                       num_classes=768,
                       heads=([2] * 2) + [1],
                       activation=F.elu,
                       feat_drop=0,
                       attn_drop=0,
                       negative_slope=0.2,
                       residual=False)
        self.layer_norm = nn.LayerNorm(torch.Size([97]))
        self.docred_label_embedding = pickle.load(
            open('DocRED_label_embedding.pkl', 'rb'))
        self.relu = nn.LeakyReLU(0.2)
        self.linear1 = nn.Linear(97, 97)
        self.linear2 = nn.Linear(97 * 2, 97)
        self.hs_lstm = nn.LSTM(input_size=768 + 97, hidden_size=768, num_layers=1, batch_first=True, dropout=0.2)
        self.ts_lstm = nn.LSTM(input_size=768 + 97, hidden_size=768, num_layers=1, batch_first=True, dropout=0.2)
        self.hs_linear = nn.Linear(768 + 97, 768)
        self.ts_linear = nn.Linear(768 + 97, 768)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert",
                                                       "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            e_emb.append(sequence_output[
                                             i, start + offset])
                            e_att.append(attention[i, :,
                                         start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0),
                                                dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def get_adjacency_matrix(self, input_ids, entity_pos, sequence_output):
        new_sequence_output = []
        mention_indexes_with_nonzero = []
        for i in range(len(entity_pos)):
            each_adjacency_matrix, each_mention_indexes_with_nonzero = self.get_each_adjmat(input_ids[i], entity_pos[i],
                                                                                            sequence_output[i])
            new_sequence_output.append(each_adjacency_matrix)
            mention_indexes_with_nonzero.append(each_mention_indexes_with_nonzero)
        return new_sequence_output, mention_indexes_with_nonzero

    def get_each_adjmat(self, each_input_id, each_entity_pos, each_sequence_output):
        each_mention_indexes_with_nonzero = []
        matrix_size = each_sequence_output.size()[
            0]
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        intra_entity_edge = np.zeros((matrix_size, matrix_size))
        for e in each_entity_pos:
            e_star_idx = []
            for start, end in e:
                if start + offset < matrix_size:
                    e_star_idx.append(start + offset)
            self.auxiliary_function_1(e_star_idx, intra_entity_edge)
            each_mention_indexes_with_nonzero.extend(e_star_idx)
        each_mention_indexes_with_nonzero = sorted(each_mention_indexes_with_nonzero)

        inter_entity_edge = np.zeros((matrix_size, matrix_size))
        e_star_all = each_mention_indexes_with_nonzero
        fullstop_idx = []
        for token_idx, token_id in enumerate(each_input_id):
            if token_id == 119:
                fullstop_idx.append(token_idx)
        fullstop_idx = sorted(fullstop_idx)
        if len(fullstop_idx) == 0:
            self.auxiliary_function_1(e_star_all, inter_entity_edge)
        elif len(fullstop_idx) == 1:
            fullstop = fullstop_idx.pop(0)
            tmp = []
            for idx, mention in enumerate(e_star_all):
                if mention < fullstop:
                    tmp.append(mention)
                else:
                    if len(e_star_all[idx:]) <= 1:
                        pass
                    else:
                        theotherentities = e_star_all[idx:]
                        self.auxiliary_function_1(theotherentities, inter_entity_edge)
            self.auxiliary_function_1(tmp, inter_entity_edge)
        else:
            while len(fullstop_idx) > 1:
                tmp = []
                period = fullstop_idx.pop(0)
                for mention_idx, mention in enumerate(e_star_all):
                    if mention < period:
                        tmp.append(mention)
                    else:
                        break
                e_star_all = e_star_all[mention_idx:]
                self.auxiliary_function_1(tmp, inter_entity_edge)
            if len(e_star_all) > 1:
                fullstop = fullstop_idx.pop(0)
                tmp = []
                for idx, mention in enumerate(e_star_all):
                    if mention < fullstop:
                        tmp.append(mention)
                    else:
                        if len(e_star_all[idx:]) <= 1:
                            pass
                        else:
                            theotherentities = e_star_all[idx:]
                            self.auxiliary_function_1(theotherentities, inter_entity_edge)
                self.auxiliary_function_1(tmp, inter_entity_edge)
                assert len(fullstop_idx) == 0
        document_edge = np.zeros((matrix_size, matrix_size))
        document_edge[0, :] = 1
        document_edge[:, 0] = 1
        A = torch.cat([torch.from_numpy(intra_entity_edge).float().unsqueeze(-1),
                       torch.from_numpy(inter_entity_edge).float().unsqueeze(-1),
                       torch.from_numpy(document_edge).float().unsqueeze(-1),
                       ], dim=-1)
        return A, each_mention_indexes_with_nonzero

    def auxiliary_function_1(self, entities, array):
        for i in entities:
            for j in entities:
                array[i, j] = 1

    def reconstruct_allthesequence(self, orig_features, transformed_features, transformed_index):
        for idx, num in enumerate(transformed_index):
            orig_features[num] = transformed_features[idx]
        return orig_features

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        label_embedding = self.gat(self.docred_label_embedding)
        hs_with_labelinfo = torch.matmul(hs, label_embedding.transpose(0, 1))
        ts_with_labelinfo = torch.matmul(ts, label_embedding.transpose(0, 1))
        hs_with_labelinfo = self.layer_norm(hs_with_labelinfo)
        ts_with_labelinfo = self.layer_norm(ts_with_labelinfo)
        logits_with_labelinfo = self.linear2(torch.cat((hs_with_labelinfo, ts_with_labelinfo), dim=1))
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs],
                                                      dim=1)))
        ts = torch.tanh(self.tail_extractor(
            torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size,
                     self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        bl = torch.cat((bl, logits_with_labelinfo), dim=1)
        logits = self.bilinear(bl)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels, threshold=self.threshold),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

    def gen_dgl_graph(self, docred_adj, t=0.05):
        nums = np.sum(docred_adj, axis=0)
        _nums = nums[:, np.newaxis]
        for i in range(len(_nums)):
            if _nums[i] > 10:
                docred_adj[i] = docred_adj[i] / _nums[i]
            else:
                docred_adj[i] = 0
        _adj = docred_adj
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1
        _adj = _adj * 0.3 / (_adj.sum(0, keepdims=True) + 1e-6)
        start_idx = []
        end_idx = []
        for i in range(97):
            for j in range(97):
                if _adj[i, j] > 0:
                    start_idx.append(i)
                    end_idx.append(j)
        _adj = dgl.graph((start_idx, end_idx),
                         num_nodes=97)
        _adj = dgl.add_self_loop(_adj)
        return _adj
