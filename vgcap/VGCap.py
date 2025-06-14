# -*- coding: utf-8 -*-
# This file contains ESR-SAN model，We propose a new image relation features representation and language prediction method with significant modifications. Specifically, we exploit the novel visual graph topology as the inductive structure-aware relation bias, which explicitly
#improves the structural relations between object regions. For the language prediction stage, we further leverage semantic relation in the Transformer decoder for non-visual word prediction. We demonstrate that explicit semantic
#relations can enhance not only visual relations in the encoder stage but also enrich semantic word generation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import sub

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from models.CaptionModel import CaptionModel
from models.AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper
from misc.utils import expand_feats
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator


    def forward(self, src, boxes, edge_mask, rela_labels_mask, tgt, weak_rela, weak_rela_mask, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, boxes, edge_mask, rela_labels_mask, src_mask), weak_rela, weak_rela_mask, src_mask,
                            tgt, tgt_mask)

    def encode(self, src, boxes, edge_mask, rela_labels_mask, src_mask):
        return self.encoder(self.src_embed(src), boxes, edge_mask, rela_labels_mask, src_mask)

    def decode(self, memory, weak_rela, weak_rela_mask, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, weak_rela, weak_rela_mask, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, boxes, edge_mask, rela_labels_mask, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, boxes, edge_mask, rela_labels_mask, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, boxes, edge_mask, rela_labels_mask, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, boxes, edge_mask, rela_labels_mask, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, weak_rela, weak_rela_mask, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, weak_rela, weak_rela_mask, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, weak_rela, weak_rela_mask, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, weak_rela, weak_rela_mask, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return_val = torch.from_numpy(subsequent_mask) == 0
    return return_val

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttention_Crosss_Relation_Decoder(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_Crosss_Relation_Decoder, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 6)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, weak_rela, weak_rela_mask=None, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        if weak_rela_mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            weak_rela_mask = weak_rela_mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value, relation_k, relation_v = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value, weak_rela, weak_rela))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention_crosss_relation(query, key, value, relation_k, relation_v, weak_rela_mask=weak_rela_mask, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention_crosss_relation(query, key, value, relation_k, relation_v, weak_rela_mask=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    qk_matmul = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    # process q and weak relation
    qr_matmul = torch.matmul(query, relation_k.transpose(-2, -1)) / math.sqrt(query.shape[-1])

    if mask is not None:
        qk_matmul = qk_matmul.masked_fill(mask == 0, -1e9)
        qr_matmul = qr_matmul.masked_fill(mask == 0, -1e9)

    p_attn_orig_visual = F.softmax(qk_matmul, dim = -1)
    p_attn_orig_relation = F.softmax(qr_matmul, dim = -1)

    if dropout is not None:
        p_attn_orig_visual = dropout(p_attn_orig_visual)
        p_attn_orig_relation = dropout(p_attn_orig_relation)

    return torch.matmul(p_attn_orig_visual, value) + torch.matmul(p_attn_orig_relation, relation_v), p_attn_orig_visual

def relative_attention_logits(query, key, relation, boxes, linear_structure):
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.

    query_r = query.permute(0, 2, 1, 3)
    q_tr_t_matmul = torch.matmul(query_r, r_t)


    keyy = key.permute(0, 2, 3, 1)
    key_tr_t_matmul = torch.matmul(relation, keyy)
    key_tr_tmatmul_t = key_tr_t_matmul.permute(0, 3, 1, 2)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 3, 1)

    # boxs process
    # boxes_t is [batch, num queries, depth, num kvs]
    boxes_t = boxes.transpose(-2, -1)
    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_boxes_t_matmul = torch.matmul(q_t, boxes_t)
    q_boxes_t_matmul = q_boxes_t_matmul.permute(0, 2, 1, 3)
    # [batch, heads, num queries, num kvs]

    # structure process
    # linear_structure is [batch, num queries, depth, num kvs]
    linear_structure = linear_structure.transpose(-2, -1)
    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_structure_t_matmul = torch.matmul(q_t, linear_structure)
    q_structure_t_matmul = q_structure_t_matmul.permute(0, 2, 1, 3)
    # [batch, heads, num queries, num kvs]

    return ((qk_matmul + q_tr_tmatmul_t + key_tr_tmatmul_t + q_boxes_t_matmul + q_structure_t_matmul) / math.sqrt(query.shape[-1]))


class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, h, d_model, semantic_embedding_size, geometry_embedding_size, structure_embedding_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.semantic_embedding_size = semantic_embedding_size
        self.geometry_embedding_size = geometry_embedding_size
        self.structure_embedding_size = structure_embedding_size
        self.rela_embed_linear_k = nn.Sequential(nn.Linear(self.semantic_embedding_size, 64),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))
        self.rela_embed_linear_v = nn.Sequential(nn.Linear(self.semantic_embedding_size, 64),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))
        self.linear_structure = nn.Sequential(nn.Linear(self.structure_embedding_size, 64),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))
        self.linears_boxes = nn.Sequential(nn.Linear(self.geometry_embedding_size, self.d_k),
                                    nn.ReLU(),
                                    nn.Dropout(0.1))
        self.Value_weight = nn.Linear(3 * 64, 2)

    def relative_attention_values(self, weight, value, relation, boxes):
        # In this version, relation vectors are shared across heads.
        # weight: [batch, heads, num queries, num kvs].
        # value: [batch, heads, num kvs, depth].
        # relation: [batch, num queries, num kvs, depth].

        # wv_matmul is [batch, heads, num queries, depth]
        wv_matmul = torch.matmul(weight, value)

        # w_t is [batch, num queries, heads, num kvs]
        w_t = weight.permute(0, 2, 1, 3)

        #   [batch, num queries, heads, num kvs]
        # * [batch, num queries, num kvs, depth]
        # = [batch, num queries, heads, depth]
        w_tr_matmul = torch.matmul(w_t, relation)

        # w_tr_matmul_t is [batch, heads, num queries, depth]
        w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

        w_boxes_matmul = torch.matmul(w_t, boxes)
        w_boxes_matmul_t = w_boxes_matmul.permute(0, 2, 1, 3)
        # Gating Mechanism
        f = torch.cat((wv_matmul, w_tr_matmul_t, w_boxes_matmul_t), dim=-1)

        tr = torch.sigmoid(self.Value_weight(f))
        tr_wgt = tr[:, :, :, 0].unsqueeze(-1)
        boxs_wgt = tr[:, :, :, 1].unsqueeze(-1)

        return wv_matmul + tr_wgt * w_tr_matmul_t + boxs_wgt * w_boxes_matmul_t

    # Adapted from The Annotated Transformer
    def attention_with_relations(self, query, key, value, relation_k, relation_v, boxes, linear_structure, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = relative_attention_logits(query, key, relation_k, boxes, linear_structure)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn_orig = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn_orig)
        return self.relative_attention_values(p_attn, value, relation_v, boxes), p_attn_orig

    def forward(self, query, key, value, boxes, edge_mask, rela_labels_mask, mask=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        boxes_embedding = utils.BoxRelationalEmbedding(boxes, self.geometry_embedding_size) #B*N*4-->B*N*N*4/64
        boxes = self.linears_boxes(boxes_embedding)           #B*N*N*4-->B*N*N*64

        relation_k = self.rela_embed_linear_k(rela_labels_mask)
        relation_v = self.rela_embed_linear_v(rela_labels_mask)
        linear_structure = self.linear_structure(edge_mask)

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn = self.attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            boxes,
            linear_structure,
           # rela_masks=rela_masks,
            mask=mask,
            dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
##########################################################################################

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

def build_embeding_layer(vocab_size, dim, drop_prob):
    embed = nn.Sequential(nn.Embedding(vocab_size, dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob))
    return embed

class VGCap(CaptionModel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, semantic_embedding_size=256,
                geometry_embedding_size=256, structure_embedding_size=256, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttentionWithRelations(h, d_model, semantic_embedding_size, geometry_embedding_size, structure_embedding_size)
        attn_decoder = MultiHeadedAttention(h, d_model)
        attn_cross_relation_decoder = MultiHeadedAttention_Crosss_Relation_Decoder(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn_decoder), c(attn_cross_relation_decoder),
                                 c(ff), dropout), N),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(VGCap, self).__init__()
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
       # d_model = self.input_encoding_size # 512

        self.semantic_embedding_size = opt.semantic_embedding_size
        self.geometry_embedding_size = opt.geometry_embedding_size
        self.structure_embedding_size = opt.structure_embedding_size
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        # self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.geometry_relation = opt.geometry_relation
        if opt.use_box_geometry:
            self.att_feat_size = self.att_feat_size + 5  # concat box position features fomula 2:(x,y,w,h,wh)
        self.sg_label_embed_size = opt.sg_label_embed_size
        self.seq_per_img = opt.seq_per_img
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0 # Schedule sampling probability
        self.max_shortest_path_distance = 101 # NOTE: Unreachable nodes are set to 101 in shortest_path_distance.pyx
        self.MAX_EMBED_ID = 9501

        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                   nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))
        self.embed_rela = build_embeding_layer(self.MAX_EMBED_ID + 1, self.semantic_embedding_size, self.drop_prob_lm)
        self.embed_structure  = build_embeding_layer(2*self.max_shortest_path_distance + 1, self.structure_embedding_size, self.drop_prob_lm)
        self.embed_weak_rela = build_embeding_layer(self.MAX_EMBED_ID + 1, self.input_encoding_size, self.drop_prob_lm)

        self.init_weights()

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(0, tgt_vocab,
            N=opt.num_layers,
            d_model=opt.input_encoding_size,
            d_ff=opt.rnn_size,
            semantic_embedding_size = self.semantic_embedding_size,
            geometry_embedding_size = self.geometry_embedding_size,
            structure_embedding_size = self.structure_embedding_size)

    def init_weights(self):
        initrange = 0.1
        self.embed_rela[0].weight.data.uniform_(-initrange, initrange)
        self.embed_structure[0].weight.data.uniform_(-initrange, initrange)
        self.embed_weak_rela[0].weight.data.uniform_(-initrange, initrange)

    def prepare_esr_features(self, sg_data, att_feats):
        rela_labels_mask = sg_data['rela_labels_mask']

        rela_labels_mask = self.embed_rela(rela_labels_mask)

        edge_mask = sg_data['obj_dis']
        edge_mask = self.embed_structure(edge_mask)
        weak_rela = sg_data['verb_labels']
        weak_rela_mask = weak_rela > 0

        weak_rela = self.embed_weak_rela(weak_rela)

        return  rela_labels_mask, att_feats, edge_mask, weak_rela, weak_rela_mask

    def prepare_core_args(self, sg_data, att_feats, att_masks, boxes, seq=None ):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        boxes = self.clip_att(boxes, att_masks)[0]
        rela_labels_mask, att_feats, edge_mask, weak_rela, weak_rela_mask = self.prepare_esr_features(sg_data, att_feats)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] = 1

            seq_mask = seq_mask.unsqueeze(-2)
            _tmp = subsequent_mask(seq.size(-1))#.to(seq_mask)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        weak_rela_mask = weak_rela_mask.unsqueeze(-2)

        core_args = [ att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, seq, att_masks, seq_mask]
        return core_args

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _forward(self, sg_data, att_feats, boxes, seq, att_masks=None):
        att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, seq, att_masks, seq_mask = self.prepare_core_args(sg_data, att_feats, att_masks, boxes, seq)

       # make seq_per_img copies of the encoded inputs:  shape: (B, ...) => (B*seq_per_image, ...)  seq, seq_mask has original B*seq_per_image,
        core_args0 = [att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, att_masks ]

        core_args1 = expand_feats(core_args0, self.seq_per_img)
        att_feats = core_args1[0]
        boxes = core_args1[1]
        edge_mask = core_args1[2]
        rela_labels_mask = core_args1[3]
        weak_rela = core_args1[4]
        weak_rela_mask = core_args1[5]
        att_masks = core_args1[6]
        out = self.model(att_feats, boxes, edge_mask, rela_labels_mask, seq, weak_rela, weak_rela_mask, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs

    def get_logprobs_state(self, it, memory, weak_rela, weak_rela_mask, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, weak_rela, weak_rela_mask, mask,
                               ys,
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        logprobs = self.model.generator(out[:, -1])

        return logprobs, [ys.unsqueeze(0)]

    def _sample_beam(self, sg_data, att_feats, boxes, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, seq, att_masks, seq_mask = self.prepare_core_args(
            sg_data, att_feats, att_masks, boxes)
       # core_args = [att_feats, seq, att_masks, seq_mask]
        memory = self.model.encode(att_feats, boxes, edge_mask, rela_labels_mask, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = memory[k:k+1].expand(*((beam_size,)+memory.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None
            tmp_weak_rela = weak_rela[k:k + 1].expand(*((beam_size,) + weak_rela.size()[1:])).contiguous()
            tmp_weak_rela_mask = weak_rela_mask[k:k + 1].expand(
                *((beam_size,) + weak_rela_mask.size()[1:])).contiguous() if weak_rela_mask is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_weak_rela, tmp_weak_rela_mask, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_weak_rela, tmp_weak_rela_mask, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        #return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, sg_data, att_feats, boxes, att_masks=None, opt={},_core_args=None):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        return_core_args = opt.get('return_core_args', True)
        expand_features = opt.get('expand_features', False)
        if beam_size > 1:
            return self._sample_beam(sg_data, att_feats, boxes, att_masks, opt)

        if _core_args is not None:
            # reuse the core_args calculated during generating sampled captions
            # when generating greedy captions for SCST,
            core_args0 = _core_args
        else:
            att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, seq, att_masks, seq_mask = self.prepare_core_args(
                sg_data, att_feats, att_masks, boxes)
            core_args0 = [att_feats, boxes, edge_mask, rela_labels_mask, weak_rela, weak_rela_mask, att_masks]

        # should be True when training (xe or scst), False when evaluation
        if expand_features:
            core_args1 = expand_feats(core_args0, self.seq_per_img)
            att_feats = core_args1[0]
            boxes = core_args1[1]
            edge_mask = core_args1[2]
            rela_labels_mask = core_args1[3]
            weak_rela = core_args1[4]
            weak_rela_mask = core_args1[5]
            att_masks = core_args1[6]
            batch_size = att_feats.size(0)
        else:
            batch_size = att_feats.size(0)

        state = None
        memory = self.model.encode(att_feats, boxes, edge_mask, rela_labels_mask, att_masks)

        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, memory, weak_rela, weak_rela_mask, att_masks, state)
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        returns = [seq, seqLogprobs]
        if return_core_args:
            returns.append(_core_args)
        return returns
