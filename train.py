import torch
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from utils import parse_args, LOG_DIR,SAVE_DIR
from parse_tree_seq import my_sememe_tree, sememe
from model.minGPT.mingpt.model import GPT

from tqdm import tqdm

from tensorboardX import SummaryWriter   

device = "cuda"

class dict_dataset(Dataset):
    '''
    将list装换为 torch dataset
    '''
    def __init__(self, dict_data):
        self.data = list(dict_data.values())

        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def my_collate_fn(collated_data,sememe2id):
    tree_seqs = []
    tree_seqs_tokenized = []
    trees = []
    for instance in collated_data:
        tree_seq = instance[0].to_tree_sequence(add_eos=True)
        tree_seqs.append(tree_seq)
        tree_seqs_tokenized.append(list(map(lambda x: sememe2id[x.__repr__()], tree_seq)))
        trees.append(instance[0])
    pad_token_id = sememe2id[sememe("PAD|PAD").__repr__()]
    
    max_seq_length = 0
    for cont in tree_seqs_tokenized:
        max_seq_length = max(max_seq_length,len(cont))
    attention_mask = torch.zeros(len(tree_seqs_tokenized), max_seq_length).long().to(device)
    for i in range(len(tree_seqs_tokenized)): #pad到最大长度
        attention_mask[i][:len(tree_seqs_tokenized[i])] = 1
        tree_seqs_tokenized[i].extend([pad_token_id]*(max_seq_length-len(tree_seqs_tokenized[i]))  )
    tree_seqs_tokenized = torch.tensor(tree_seqs_tokenized).long().to(device)
    return trees, tree_seqs,{"input_ids":tree_seqs_tokenized,"attention_mask":attention_mask}

def get_model(sememes, args):
    '''
    给定所有的可见sememe，返回model
    '''
    vocab_size = len(sememes) #所有数据集里的义原加上，需要包含 PADtoken和BACK token
    model_config = GPT.get_default_config()
    model_config.model_type = "openai-gpt" #110M


    model_config.embedding_dim = 200
    model_config.vocab_size = vocab_size
    model_config.block_size = 1024

    model = GPT(model_config)
    if args.pretrained: #加载预训练 sememe embedding权重
        pass
    optimizer = model.configure_optimizers(learning_rate=args.learning_rate, weight_decay=args.weight_decay, betas=args.betas)
    return model, optimizer 

@torch.inference_mode()
def evals(args,steps,summary_writer, model, test_data,sememe2id,id2sememe):

    model.eval()
    test_loader = DataLoader(test_data,batch_size=args.eval_batch_size,collate_fn=lambda x: my_collate_fn(x,sememe2id=sememe2id) )
    loss_fn = CrossEntropyLoss(reduce=False)
    losses = []
    for trees, tree_seqs,tree_seqs_tokenized in tqdm(test_loader,total=len(test_loader)):
        logits, loss = model(**tree_seqs_tokenized) 
        loss = loss_fn(logits[:,:-1,:].reshape(-1,logits.shape[-1]), tree_seqs_tokenized["input_ids"][:,1:].reshape(-1))
        loss = (loss * tree_seqs_tokenized["attention_mask"][:,1:].reshape(-1)).sum() / (tree_seqs_tokenized["attention_mask"][:,1:].sum()) #对所
        losses.append(loss.item())

        start_ids = tree_seqs_tokenized["input_ids"][:,0:1]
        outputs = model.generate(
            idx = start_ids, 
            max_new_tokens = 30, 
            temperature=1.0, 
            do_sample=False, 
            top_k=1, #greedy
        )

        output_sememes_batch  = []
        for i in range(len(trees)):
            trees[i].print()
            print(tree_seqs[i])
            output_sememes = list(map(lambda x: id2sememe[x], outputs[i].cpu().tolist()))
            print(output_sememes)
            output_sememes_batch.append(output_sememes)
        exit()
    mean_loss = np.mean(np.array(losses))
    if summary_writer != None:
        summary_writer.add_scalars("loss",{"test": mean_loss},global_step=steps)

    model.train()




def checkpoint_model(args, model, step):
    save_path = f'{SAVE_DIR}/{args.exp_name}/{step}.pt'
    torch.save(model.state_dict(),save_path)
    return save_path

def train(args):
    traindata = torch.load(args.train_set_path)
    train_dataset = dict_dataset(traindata)
    testdata = torch.load(args.test_set_path)
    test_dataset = dict_dataset(testdata)

    sememes = torch.load(args.sememe_data_path)
    sememe2id = {}
    id2sememe = []
    for k,key in enumerate(sememes.keys()):
        sememe2id[key] = k  
        id2sememe.append(key)


    model, optimizer = get_model(sememes, args)
    model = model.to(device)


    if args.ckpt_path != "":
        model.load_state_dict(torch.load(args.ckpt_path))
    if args.test_only:
        evals(args,0,None, model, test_dataset,sememe2id,id2sememe)
        return 

    lr_scheduler = CosineAnnealingLR(optimizer,args.max_epoch*len(train_dataset)//args.train_batch_size,eta_min=0)

    loss_fn = CrossEntropyLoss(reduce=False)

    summary_writer = SummaryWriter(os.path.join(LOG_DIR,args.exp_name))

    steps = 0
    for epoch in range(args.max_epoch):
        train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size,collate_fn=lambda x: my_collate_fn(x,sememe2id=sememe2id) )

        for trees, tree_seqs,tree_seqs_tokenized in tqdm(train_loader,total=len(train_loader)):
            # print(tree_seqs)
            # print(tree_seqs_tokenized)
            # trees[0].print()
            # exit()
            logits, loss = model(**tree_seqs_tokenized) 
            loss = loss_fn(logits[:,:-1,:].reshape(-1,logits.shape[-1]), tree_seqs_tokenized["input_ids"][:,1:].reshape(-1))
            loss = (loss * tree_seqs_tokenized["attention_mask"][:,1:].reshape(-1)).sum() / (tree_seqs_tokenized["attention_mask"][:,1:].sum()) #对所有有效位置取平均
            # print(loss  )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            summary_writer.add_scalars("loss",{"train": loss.item()},global_step=steps)
            summary_writer.add_scalar("grad_norm",grad_norm,global_step=steps)
            # print(lr_scheduler.get_last_lr())
            summary_writer.add_scalar("lr",lr_scheduler.get_last_lr()[0],global_step=steps)


            if (steps + 1) % args.eval_steps == 0:
                evals(args,steps+1,summary_writer, model, test_dataset,sememe2id,id2sememe)
            
            if (steps + 1) % args.save_steps == 0:
                checkpoint_model(args, model, steps+1)
            steps += 1


def main():
    args = parse_args()
    print(args)

    os.makedirs(os.path.join(SAVE_DIR,args.exp_name),exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR,args.exp_name),exist_ok=True)
    # print(args.sememe_data_path)
    train(args)


if __name__ == "__main__":
    # x = [1]
    # x.extend([""]*0)
    # print(x)
    main()