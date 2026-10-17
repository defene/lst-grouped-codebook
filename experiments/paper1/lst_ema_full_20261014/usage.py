"""Streaming counts; bounded memory even for the complete validation set."""
import numpy as np

class Usage:
    def __init__(self,c):
        self.grouped=c['model']['kind']=='grouped_g8k256'
        self.scales=c['model'].get('scales',[16]) if self.grouped else [16]
        self.counts=np.zeros((c['model'].get('groups',1) if self.grouped else 1,len(self.scales),c['model']['codebook_size']),dtype=np.int64)
    def add(self,indices):
        ids=indices.detach().cpu().numpy()
        if not self.grouped:ids=ids.reshape(len(ids),1,-1)
        start=0
        for si,scale in enumerate(self.scales):
            for g in range(len(self.counts)):
                self.counts[g,si]+=np.bincount(ids[:,g,start:start+scale*scale].ravel(),minlength=self.counts.shape[-1])
            start+=scale*scale
        if start!=ids.shape[-1]:raise ValueError('Unexpected quantizer layout')
    def result(self):
        def summarize(counts):
            p=counts/counts.sum()
            return dict(used_codes=int((counts>0).sum()),perplexity=float(np.exp(-(p*np.log(p.clip(1e-30))).sum())),
                        top1_fraction=float(p.max()),token_count=int(counts.sum()),counts=counts.tolist())
        return dict(scope='All transmitted latent positions; independent groups, shared across scales',
            groups=[dict(group=g,overall=summarize(counts.sum(0)),
                scales=[dict(scale=s,**summarize(row)) for s,row in zip(self.scales,counts)]) for g,counts in enumerate(self.counts)])
