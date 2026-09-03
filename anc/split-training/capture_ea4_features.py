#!/usr/bin/env python3
"""Capture document-level input-boundary features from an E-A4 checkpoint."""
import argparse, hashlib, json, statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--corpus-file",required=True); ap.add_argument("--condition",choices=["split_ft","fedavg"],required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--max-docs",type=int,default=80)
    ap.add_argument("--seq-len",type=int,default=128); ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from split_trainer import make_layer_kwargs, run_layer_stack
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
                                                device_map=args.device,
                                                attn_implementation="eager").eval()
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    if ck.get("condition") != args.condition: raise ValueError("checkpoint condition mismatch")
    depth=int(ck["split_after"]); embed=model.model.embed_tokens
    head=torch.nn.ModuleList(list(model.model.layers[:depth+1]))
    embed.load_state_dict(ck["embed"]); head.load_state_dict(ck["head"])
    docs=[x.strip() for x in Path(args.corpus_file).read_text().splitlines() if len(x.strip())>20]
    docs=docs[:min(len(docs),args.max_docs)]
    if len(docs)<20: raise ValueError("need at least 20 documents")
    member_cut=len(docs)//2; median_len=statistics.median(map(len,docs))
    gen=torch.Generator(device=args.device).manual_seed(2026)
    hidden=model.config.hidden_size
    projection=torch.randn(hidden,16,generator=gen,device=args.device)/hidden**0.5
    ns=argparse.Namespace(attn_impl="eager")
    records=[]
    with torch.no_grad():
        for i,doc in enumerate(docs):
            ids=tok(doc,return_tensors="pt",truncation=True,max_length=args.seq_len).input_ids.to(args.device)
            pos=torch.arange(ids.shape[1],device=args.device).unsqueeze(0)
            h=embed(ids); rotary=model.model.rotary_emb
            lk=make_layer_kwargs(rotary,h,pos,ns); h=run_layer_stack(head,h,lk).float()[0]
            projected=h @ projection
            feats=torch.cat([projected.mean(0),projected.std(0,unbiased=False),
                             h.norm(dim=-1).mean().view(1),h.norm(dim=-1).std(unbiased=False).view(1)])
            records.append({"document_id":hashlib.sha256(doc.encode()).hexdigest(),
                            "condition":args.condition,"membership":int(i<member_cut),
                            "property":int(len(doc)>median_len),
                            "features":[round(float(v),7) for v in feats.cpu()]})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    print(f"Wrote {len(records)} records to {out}")
if __name__=="__main__": main()
