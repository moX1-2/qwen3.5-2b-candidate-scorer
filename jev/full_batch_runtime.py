"""完整候选跨题训练窗口，OOM 时清梯度并重跑完整窗口。"""
import gc
import torch
import torch.nn.functional as F
from shared_prefix_training import encode_branches, score_question

def full_window(model, head, records, processor, root, max_length, template, log):
    device=next(head.parameters()).device
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    def attempt(conservative):
        loss_sum=torch.zeros((),device=device,dtype=torch.float64)
        images=0;texts=0;cursor=0
        while cursor<len(records):
            r=records[cursor]
            if r.get('image') or r.get('image_path'):
                model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                encoded=encode_branches(processor,r,root,device,max_length,template)
                scores=score_question(model,head,encoded,shared=False)
                loss=F.cross_entropy(scores[None],torch.tensor([r['gold_index']],device=device))
                (loss/len(records)).backward();loss_sum+=loss.detach()
                images+=1;cursor+=1
                del loss,scores,encoded
                continue
            chunks=[];golds=[];flat=[];maximum=0
            limit=1 if conservative else 4
            while cursor+len(chunks)<len(records) and len(chunks)<limit:
                item=records[cursor+len(chunks)]
                if item.get('image') or item.get('image_path'):break
                encoded=encode_branches(processor,item,root,device,max_length,template)
                longest=max(x['input_ids'].shape[-1] for x in encoded)
                projected=max(maximum,longest)*(len(flat)+len(encoded))
                if chunks and projected>10000:break
                chunks.append(encoded);flat.extend(encoded);golds.append(item['gold_index']);maximum=max(maximum,longest)
            free=torch.cuda.mem_get_info()[0]/2**30 if torch.cuda.is_available() else 0
            checkpointing=conservative or free<55 or maximum>768
            if checkpointing:model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:model.language_model.gradient_checkpointing_disable()
            counts=[len(c) for c in chunks]
            ids=torch.cat([F.pad(x['input_ids'],(0,maximum-x['input_ids'].shape[-1]),value=processor.tokenizer.pad_token_id) for x in flat])
            mask=torch.cat([F.pad(x['attention_mask'],(0,maximum-x['input_ids'].shape[-1]),value=0) for x in flat])
            out=model.language_model(input_ids=ids,attention_mask=mask,use_cache=False)
            hidden=out.last_hidden_state[torch.arange(len(flat),device=device),mask.sum(-1)-1]
            scores=head(hidden.float()).squeeze(-1).split(counts)
            loss=sum(F.cross_entropy(s[None],torch.tensor([g],device=device)) for s,g in zip(scores,golds))
            (loss/len(records)).backward();loss_sum+=loss.detach()
            texts+=len(chunks);cursor+=len(chunks)
            del loss,scores,out,hidden,ids,mask,flat,chunks,encoded
        return loss_sum,texts,images
    try:
        return attempt(False)
    except torch.cuda.OutOfMemoryError:
        pass
    # 离开异常处理作用域后，失败前向的局部变量与计算图才能释放。
    model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
    log({'event':'oom_retry','strategy':'full_batch1_checkpoint','window_examples':len(records)})
    return attempt(True)
