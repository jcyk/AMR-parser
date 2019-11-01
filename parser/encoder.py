import torch
from torch import nn
import torch.nn.functional as F

from transformer import Embedding

class WordEncoder(nn.Module):
    def __init__(self, vocabs, char_dim, word_dim, pos_dim, ner_dim,
        embed_dim, filters, char2word_dim, dropout,
        pretrained_word_embed = None):
        super(WordEncoder, self).__init__()
        self.char_embed = Embedding(vocabs['word_char'].size, char_dim, padding_idx=vocabs['word_char'].padding_idx)
        self.char2word = CNNEncoder(filters, char_dim, char2word_dim)
        self.lem_embed = Embedding(vocabs['lem'].size, word_dim, padding_idx=vocabs['lem'].padding_idx)

        if pos_dim > 0:
            self.pos_embed = Embedding(vocabs['pos'].size, pos_dim, padding_idx=vocabs['pos'].padding_idx)
        else:
            self.pos_embed = None
        if ner_dim > 0:
            self.ner_embed = Embedding(vocabs['ner'].size, ner_dim, padding_idx=vocabs['ner'].padding_idx)
        else:
            self.ner_embed = None

        tot_dim = word_dim + pos_dim + ner_dim + char2word_dim
        
        self.pretrained_word_embed = pretrained_word_embed
        if self.pretrained_word_embed is not None:
            tot_dim += self.pretrained_word_embed.embedding_dim
        
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, char_input, tok_input, lem_input, pos_input, ner_input):
        # char: seq_len x bsz x word_len
        # word, pos, ner: seq_len x bsz
        seq_len, bsz, _ = char_input.size()
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        char_repr = self.char2word(char_repr).view(seq_len, bsz, -1)

        if self.pretrained_word_embed is not None:
            lem_repr = self.lem_embed(lem_input)
            tok_repr = self.pretrained_word_embed(tok_input)
            reprs = [char_repr, lem_repr, tok_repr]  
        else:
            lem_repr = self.lem_embed(lem_input)
            reprs = [char_repr, lem_repr]

        if self.pos_embed is not None:
            pos_repr = self.pos_embed(pos_input)
            reprs.append(pos_repr)
        
        if self.ner_embed is not None:
            ner_repr = self.ner_embed(ner_input)
            reprs.append(ner_repr)

        word = F.dropout(torch.cat(reprs, -1), p=self.dropout, training=self.training)
        word = self.out_proj(word)
        return word

class ConceptEncoder(nn.Module):
    def __init__(self, vocabs, char_dim, concept_dim, embed_dim, filters, char2concept_dim, dropout):
        super(ConceptEncoder, self).__init__()
        self.char_embed = Embedding(vocabs['concept_char'].size, char_dim, padding_idx=vocabs['concept_char'].padding_idx)
        self.concept_embed = Embedding(vocabs['concept'].size, concept_dim, padding_idx=vocabs['concept'].padding_idx)
        self.char2concept = CNNEncoder(filters, char_dim, char2concept_dim)
        self.vocabs = vocabs
        tot_dim = char2concept_dim + concept_dim
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        self.char_dim = char_dim
        self.concept_dim = concept_dim
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, char_input, concept_input):

        seq_len, bsz, _ = char_input.size()
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        char_repr = self.char2concept(char_repr).view(seq_len, bsz, -1)
        concept_repr = self.concept_embed(concept_input)

        concept = F.dropout(torch.cat([char_repr,concept_repr], -1), p=self.dropout, training=self.training)
        concept = self.out_proj(concept)
        return concept


class CNNEncoder(nn.Module):
    def __init__(self, filters, input_dim, output_dim, highway_layers=1):
        super(CNNEncoder, self).__init__()
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(nn.Conv1d(input_dim, out_c, kernel_size=width))
        final_dim = sum(f[1] for f in filters)
        self.highway = Highway(final_dim, highway_layers)
        self.out_proj = nn.Linear(final_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, input):
        # input: batch_size x seq_len x input_dim
        x  = input.transpose(1, 2)
        conv_result = []
        for i, conv in enumerate(self.convolutions):
            y = conv(x)
            y, _ = torch.max(y, -1)
            y = F.relu(y)
            conv_result.append(y)

        conv_result = torch.cat(conv_result, dim=-1)
        conv_result = self.highway(conv_result)
        return self.out_proj(conv_result) #  batch_size x output_dim

class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x
