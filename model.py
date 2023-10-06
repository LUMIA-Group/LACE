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
        self.linear2 = nn.Linear(97 * 2, 97)

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):
        # Note that for GloVe embedding, leverage the following similar operations:
        # tok_emb = torch.cat([word_emb, ner_embs])
        # word_emb = pack_padded_sequence(tok_emb)
        # packed_output, (hidden, _) = self.lstm(word_emb)
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

    def gen_dgl_graph(self, docred_adj, t=0.05, p=0.3):
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
        _adj = _adj * p / (_adj.sum(0, keepdims=True) - 1 + 1e-6)
        row, col = np.diag_indices_from(_adj)
        _adj[row, col] = 1.0 - p
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
