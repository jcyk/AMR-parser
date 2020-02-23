import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import Vocab, DataLoader,load_pretrained_word_embed, DUM, END, CLS, NIL
from parser import Parser
from work import show_progress
from extract import LexicalMap
from adam import AdamWeightDecayOptimizer
from utils import move_to_device
import argparse, os
import random
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok_vocab', type=str)
    parser.add_argument('--lem_vocab', type=str)
    parser.add_argument('--pos_vocab', type=str)
    parser.add_argument('--ner_vocab', type=str)
    parser.add_argument('--concept_vocab', type=str)
    parser.add_argument('--predictable_concept_vocab', type=str)
    parser.add_argument('--rel_vocab', type=str)
    parser.add_argument('--word_char_vocab', type=str)
    parser.add_argument('--concept_char_vocab', type=str)
    parser.add_argument('--lexical_mapping', type=str)
    parser.add_argument('--pretrained_word_embed', type=str, default=None)

    parser.add_argument('--word_char_dim', type=int)
    parser.add_argument('--word_dim', type=int)
    parser.add_argument('--pos_dim', type=int)
    parser.add_argument('--ner_dim', type=int)
    parser.add_argument('--concept_char_dim', type=int)
    parser.add_argument('--concept_dim', type=int)
    parser.add_argument('--rel_dim', type=int)

    parser.add_argument('--cnn_filters', type=int, nargs = '+')
    parser.add_argument('--char2word_dim', type=int)
    parser.add_argument('--char2concept_dim', type=int)


    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--snt_layers', type=int)
    parser.add_argument('--graph_layers', type=int)
    parser.add_argument('--inference_layers', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--unk_rate', type=float)


    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batches_per_update', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--dev_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)


    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    return parser.parse_args()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def data_proc(data, queue):
    while True:
        for x in data:
            queue.put(x)
        queue.put('EPOCHDONE')

def update_lr(optimizer, embed_size, steps, warmup_steps):
    for param_group in optimizer.param_groups:
        param_group['lr'] = embed_size**-0.5 * min(steps**-0.5, steps*(warmup_steps**-1.5))

def main(args, local_rank):
    vocabs = dict()
    vocabs['tok'] = Vocab(args.tok_vocab, 5, [CLS])
    vocabs['lem'] = Vocab(args.lem_vocab, 5, [CLS])
    vocabs['pos'] = Vocab(args.pos_vocab, 5, [CLS])
    vocabs['ner'] = Vocab(args.ner_vocab, 5, [CLS])
    vocabs['predictable_concept'] = Vocab(args.predictable_concept_vocab, 10, [DUM, END])
    vocabs['concept'] = Vocab(args.concept_vocab, 5, [DUM, END])
    vocabs['rel'] = Vocab(args.rel_vocab, 50, [NIL])
    vocabs['word_char'] = Vocab(args.word_char_vocab, 100, [CLS, END])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [CLS, END])
    lexical_mapping = LexicalMap(args.lexical_mapping)
    if args.pretrained_word_embed is not None:
        vocab, pretrained_embs = load_pretrained_word_embed(args.pretrained_word_embed)
        vocabs['glove'] = vocab
    else:
        pretrained_embs = None

    for name in vocabs:
        print ((name, vocabs[name].size))

    torch.manual_seed(19940117)
    torch.cuda.manual_seed_all(19940117)
    random.seed(19940117)
    device = torch.device('cuda', local_rank)
    model = Parser(vocabs,
            args.word_char_dim, args.word_dim, args.pos_dim, args.ner_dim,
            args.concept_char_dim, args.concept_dim,
            args.cnn_filters, args.char2word_dim, args.char2concept_dim,
            args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
            args.snt_layers, args.graph_layers, args.inference_layers, args.rel_dim,
            pretrained_embs, device=device)
    
    if args.world_size > 1:
        torch.manual_seed(19940117 + dist.get_rank())
        torch.cuda.manual_seed_all(19940117 + dist.get_rank())
        random.seed(19940117+dist.get_rank())

    model = model.cuda(local_rank)
    train_data = DataLoader(vocabs, lexical_mapping, args.train_data, args.train_batch_size, for_train=True)
    dev_data = DataLoader(vocabs, lexical_mapping, args.dev_data, args.dev_batch_size, for_train=True)
    train_data.set_unk_rate(args.unk_rate)

    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay':1e-4},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    optimizer = AdamWeightDecayOptimizer(grouped_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-6)

    batches_acm, loss_acm, concept_loss_acm, arc_loss_acm, rel_loss_acm = 0, 0, 0, 0, 0
    #model.load_state_dict(torch.load('./ckpt/epoch297_batch49999')['model'])
    discarded_batches_acm = 0
    queue = mp.Queue(10)
    train_data_generator = mp.Process(target=data_proc, args=(train_data, queue)) 
    train_data_generator.start()

    used_batches = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        batches_acm = ckpt['batches_acm']
        del ckpt

    model.train()
    epoch = 0
    while True:
        batch = queue.get()
        if isinstance(batch, str):
            epoch += 1
            print ('epoch', epoch, 'done', 'batches', batches_acm)
        else:
            batch = move_to_device(batch, model.device)
            concept_loss, arc_loss, rel_loss = model(batch)
            loss = (concept_loss + arc_loss + rel_loss) / args.batches_per_update
            loss_value = loss.item()
            concept_loss_value = concept_loss.item()
            arc_loss_value = arc_loss.item()
            rel_loss_value = rel_loss.item()
            if batches_acm > args.warmup_steps and arc_loss_value > 5.*(arc_loss_acm /batches_acm):
                discarded_batches_acm += 1
                print ('abnormal', concept_loss.item(), arc_loss.item(), rel_loss.item())
                continue
            loss_acm += loss_value
            concept_loss_acm += concept_loss_value
            arc_loss_acm += arc_loss_value
            rel_loss_acm += rel_loss_value
            loss.backward()

            used_batches += 1
            if not (used_batches % args.batches_per_update == -1 % args.batches_per_update):
                continue
            batches_acm += 1

            if args.world_size > 1:
                average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            update_lr(optimizer, args.embed_dim, batches_acm, args.warmup_steps)
            optimizer.step()
            optimizer.zero_grad()
            if args.world_size == 1 or (dist.get_rank() == 0):
                if batches_acm % args.print_every == -1 % args.print_every:
                    print ('Train Epoch %d, Batch %d, Discarded Batch %d, conc_loss %.3f, arc_loss %.3f, rel_loss %.3f'%(epoch, batches_acm, discarded_batches_acm, concept_loss_acm/batches_acm, arc_loss_acm/batches_acm, rel_loss_acm/batches_acm))
                    model.train()
                
                if batches_acm % args.eval_every == -1 % args.eval_every:
                    model.eval() 
                    torch.save({'args':args, 
                                'model':model.state_dict(),
                                'batches_acm': batches_acm,
                                'optimizer': optimizer.state_dict()}, '%s/epoch%d_batch%d'%(args.ckpt, epoch, batches_acm))
                    model.train()

def init_processes(args, local_rank, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    main(args, local_rank)

if __name__ == "__main__":
    args = parse_config()
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    assert len(args.cnn_filters)%2 == 0
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))

    if args.world_size == 1:
        main(args, 0)
        exit(0)
    args.train_batch_size = args.train_batch_size
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    never_stop = True
    while never_stop:
        never_stop = True
