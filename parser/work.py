import torch

from data import Vocab,  DataLoader, load_pretrained_word_embed, DUM, END, CLS, NIL
from parser import Parser
from postprocess import PostProcessor
from extract import LexicalMap
from utils import move_to_device

import argparse, os

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--max_time_step', type=int)
    parser.add_argument('--output_suffix', type=str)
    parser.add_argument('--sense_table', type=str)
    parser.add_argument('--wiki_table', type=str)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()

def show_progress(model, dev_data):
    model.eval()
    loss_acm = 0.
    for batch in dev_data:
        batch = move_to_device(batch, model.device)
        concept_loss, arc_loss, rel_loss = model(batch)
        loss = concept_loss + arc_loss + rel_loss
        loss_acm += loss.item()
    print ('total loss', loss_acm)
    return loss_acm

def parse_batch(model, batch, beam_size, alpha, max_time_step):
    res = dict()
    concept_batch = []
    relation_batch = []
    beams = model.work(batch, beam_size, max_time_step)
    score_batch = []
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_concept = [token for token in best_hyp.seq[1:-1]]
        predicted_rel = []
        for i in range(len(predicted_concept)):
            if i == 0:
                continue
            arc = best_hyp.state_dict['arc_ll%d'%i].squeeze_().exp_()[1:]# head_len
            rel = best_hyp.state_dict['rel_ll%d'%i].squeeze_().exp_()[1:,:] #head_len x vocab
            for head_id, (arc_prob, rel_prob) in enumerate(zip(arc.tolist(), rel.tolist())):
                predicted_rel.append((i, head_id, arc_prob, rel_prob))
        concept_batch.append(predicted_concept)
        score_batch.append(best_hyp.score)
        relation_batch.append(predicted_rel)
    res['concept'] = concept_batch
    res['score'] = score_batch
    res['relation'] = relation_batch
    return res

if __name__ == "__main__":

    args = parse_config()

    test_models = []
    if os.path.isdir(args.load_path):
        for file in os.listdir(args.load_path):
            fname = os.path.join(args.load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']
    else:
        test_models.append(args.load_path)
        model_args = torch.load(args.load_path)['args']

    vocabs = dict()

    vocabs['tok'] = Vocab(model_args.tok_vocab, 5, [CLS])
    vocabs['lem'] = Vocab(model_args.lem_vocab, 5, [CLS])
    vocabs['pos'] = Vocab(model_args.pos_vocab, 5, [CLS])
    vocabs['ner'] = Vocab(model_args.ner_vocab, 5, [CLS])
    vocabs['predictable_concept'] = Vocab(model_args.predictable_concept_vocab, 10, [DUM, END])
    vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [DUM, END])
    vocabs['rel'] = Vocab(model_args.rel_vocab, 50, [NIL])
    vocabs['word_char'] = Vocab(model_args.word_char_vocab, 100, [CLS, END])
    vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [CLS, END])
    lexical_mapping = LexicalMap(model_args.lexical_mapping)
    if model_args.pretrained_word_embed is not None:
        vocab, pretrained_embs = load_pretrained_word_embed(model_args.pretrained_word_embed)
        vocabs['glove'] = vocab
    else:
        pretrained_embs = None


    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = Parser(vocabs,
            model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,
            model_args.concept_char_dim, model_args.concept_dim,
            model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
            model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
            pretrained_word_embed=pretrained_embs, device=device)

    test_data = DataLoader(vocabs, lexical_mapping, args.test_data, args.test_batch_size, for_train=True)
    another_test_data = DataLoader(vocabs, lexical_mapping, args.test_data, args.test_batch_size, for_train=False)
    for test_model in test_models:
        print (test_model)
        model.load_state_dict(torch.load(test_model)['model'])
        model = model.cuda()
        model.eval()

        loss = show_progress(model, test_data)

        tot = 0
        pp = PostProcessor(args.sense_table, args.wiki_table, vocabs['rel']) 
        with open(test_model+args.output_suffix, 'w') as fo:
            for batch in another_test_data:
                batch = move_to_device(batch, model.device)
                res = parse_batch(model, batch, args.beam_size, args.alpha, args.max_time_step)
                for concept, relation, score in zip(res['concept'], res['relation'], res['score']):
                    fo.write('# ::conc '+ ' '.join(concept)+'\n')
                    fo.write('# ::score %.6f\n'%score)
                    fo.write(pp.postprocess(concept, relation)+'\n\n')
                    tot += 1
        print ('write down %d amrs'%tot)
