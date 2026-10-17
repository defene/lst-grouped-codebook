"""Deferred EMA for single-scale VQ and shared grouped multiscale dictionaries."""
import torch
from torch import nn
from torch.nn import functional as F
from eo_denoise.var_ops import interpolate
from experiments.paper1.lst_restoration_lossfix_20261014.quantizers import nearest_indices, straight_through


@torch.no_grad()
def kmeans(values, size, iterations=10):
    if not len(values): raise ValueError('Empty initialization pool')
    # Deterministic data sampling, no consumption of training RNG.
    take=torch.linspace(0,len(values)-1,size,device=values.device).long()
    centers=values[take].clone()
    for _ in range(iterations):
        ids=nearest_indices(values,centers)
        counts=torch.bincount(ids,minlength=size).float()
        sums=torch.zeros_like(centers).index_add_(0,ids,values)
        centers=torch.where((counts>0)[:,None],sums/counts.clamp_min(1)[:,None],centers)
    return centers


class DeferredEMA(nn.Module):
    def __init__(self, embedding, maintenance=False, decay=.99):
        super().__init__()
        # The owner registers the embedding; avoid duplicate state-dict paths.
        object.__setattr__(self,'embedding',embedding)
        embedding.weight.requires_grad_(False)
        self.size,self.dim=embedding.weight.shape
        self.maintenance,self.decay=maintenance,decay
        self.register_buffer('mass',torch.zeros(self.size))
        self.register_buffer('sums',torch.zeros_like(embedding.weight))
        self.register_buffer('frequency',torch.zeros(self.size))
        self.register_buffer('steps',torch.tensor(0,dtype=torch.long))
        self.register_buffer('reset_count',torch.tensor(0,dtype=torch.long))
        self.register_buffer('initialized',torch.tensor(not maintenance))
        self.register_buffer('pending_mass',torch.zeros(self.size),persistent=False)
        self.register_buffer('pending_sums',torch.zeros_like(embedding.weight),persistent=False)
        self.pool=[];self.collect=False

    @torch.no_grad()
    def initialize(self,values):
        if not bool(self.initialized):
            centers=kmeans(values.float(),self.size)
            self.embedding.weight.copy_(centers)
            self.initialized.fill_(True)

    def begin(self):
        self.pending_mass.zero_();self.pending_sums.zero_();self.pool=[];self.collect=True

    @torch.no_grad()
    def accumulate(self,values,ids):
        if not self.collect:return
        self.pending_mass.add_(torch.bincount(ids,minlength=self.size))
        self.pending_sums.index_add_(0,ids,values.detach().float())
        # Bounded candidate pool, spanning each microbatch/scale deterministically.
        n=min(128,len(values))
        self.pool.append(values.detach()[torch.linspace(0,len(values)-1,n,device=values.device).long()].clone())

    @torch.no_grad()
    def commit(self):
        total=self.pending_mass.sum()
        if not self.collect or total<=0:raise RuntimeError('No complete EMA update collected')
        d=self.decay
        self.mass.mul_(d).add_(self.pending_mass,alpha=1-d)
        self.sums.mul_(d).add_(self.pending_sums,alpha=1-d)
        self.frequency.mul_(d).add_(self.pending_mass/total,alpha=1-d)
        used=self.mass>1e-8
        updated=self.sums/self.mass.clamp_min(1e-8)[:,None]
        self.embedding.weight[used]=updated[used]
        self.steps.add_(1)
        # Bias-corrected relative usage; not SoundStream's absolute audio count.
        # Sixteen-update cadence and grace prevent constant rewrites.
        if self.maintenance and self.steps>=16 and int(self.steps)%16==0:
            corrected=self.frequency/(1-d**int(self.steps))
            dead=corrected < (.01/self.size)
            number=int(dead.sum())
            if number:
                pool=torch.cat(self.pool)
                take=torch.linspace(0,len(pool)-1,number,device=pool.device).long()
                replacement=pool[take]
                self.embedding.weight[dead]=replacement
                prior=total/self.size
                self.mass[dead]=prior
                self.sums[dead]=replacement*prior
                self.frequency[dead]=(1-d**int(self.steps))/self.size
                self.reset_count.add_(number)
        self.collect=False;self.pool=[]


class EMAVQ(nn.Module):
    def __init__(self,original,maintenance=False):
        super().__init__()
        self.embedding=original.embedding
        self.codebook_size=original.codebook_size
        self.dim=original.dim;self.commitment=original.commitment
        self.ema=DeferredEMA(self.embedding,maintenance)

    def decode_indices(self,ids):
        return self.embedding(ids.long()).float().permute(0,3,1,2).contiguous()

    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=x.float();b,c,h,w=x.shape
            flat=x.permute(0,2,3,1).reshape(-1,c)
            ids=nearest_indices(flat,self.embedding.weight).reshape(b,h,w)
            codes=self.decode_indices(ids)
            if self.training:self.ema.accumulate(flat,ids.flatten())
            commitment=F.mse_loss(x,codes.detach())
            counts=torch.bincount(ids.flatten(),minlength=self.codebook_size).float();p=counts/counts.sum()
            diagnostics=dict(quantization_mse=commitment,active_codes=(counts>0).sum(),
                perplexity=torch.exp(-(p*p.clamp_min(1e-20).log()).sum()),
                latent_rms=x.square().mean().sqrt(),codebook_rms=self.embedding.weight.square().mean().sqrt())
            return dict(quantized=straight_through(x,codes),indices=ids,loss=self.commitment*commitment,
                        diagnostics={k:v.detach() for k,v in diagnostics.items()})


class EMAGrouped(nn.Module):
    def __init__(self,original,maintenance=False):
        super().__init__()
        self.grouped=original.grouped
        self.scales=original.scales;self.groups=original.groups;self.codebook_size=original.codebook_size
        self.emas=nn.ModuleList([DeferredEMA(e,maintenance) for e in self.grouped.embeddings])

    def decode_indices(self,ids):return self.grouped.decode(ids)

    def forward(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            z=z.float();b,c,h,w=z.shape
            core=self.grouped
            rest,total=z.detach().clone(),torch.zeros_like(z)
            loss=z.new_zeros(());indices=[];diagnostics={}
            for si,pn in enumerate(self.scales):
                low=interpolate(rest,(pn,pn),'area') if pn!=h else rest
                ids=[]
                for g,(part,embedding,ema) in enumerate(zip(low.chunk(self.groups,1),core.embeddings,self.emas)):
                    flat=part.permute(0,2,3,1).reshape(-1,c//self.groups)
                    assignment=nearest_indices(flat,embedding.weight)
                    ids.append(assignment.reshape(b,pn*pn))
                    if self.training:ema.accumulate(flat,assignment)
                idx=torch.stack(ids,1);contribution=core.contribution(idx,si)
                total=total+contribution;rest=rest-contribution.detach();indices.append(idx)
                if self.training:
                    # Embeddings are EMA-only. Retain the native Phi fitting
                    # term because Phi is a learned full-channel residual map.
                    loss=loss+core.beta*F.mse_loss(total.detach(),z)+F.mse_loss(total,z.detach())
            ids=torch.cat(indices,2)
            diagnostics['quantization_mse']=F.mse_loss(z,total).detach()
            for g in range(self.groups):
                counts=torch.bincount(ids[:,g].flatten(),minlength=self.codebook_size).float();p=counts/counts.sum()
                diagnostics[f'group{g}/active_codes']=(counts>0).sum()
                diagnostics[f'group{g}/perplexity']=torch.exp(-(p*p.clamp_min(1e-20).log()).sum())
            return dict(quantized=straight_through(z,total),indices=ids,loss=loss/len(self.scales),
                        diagnostics={k:v.detach() for k,v in diagnostics.items()})


def controllers(model):
    return [m for m in model.modules() if isinstance(m,DeferredEMA)]


@torch.no_grad()
def initialize_from_latents(q,latent):
    """One training calibration batch. No model weights/EMA usage updated."""
    if isinstance(q,EMAVQ):
        q.ema.initialize(latent.permute(0,2,3,1).reshape(-1,q.dim))
    else:
        # Shared dictionaries seeded from all spatial locations of the first
        # unquantized training latent, not sequentially reinitialized per scale.
        for part,ema in zip(latent.chunk(q.groups,1),q.emas):
            ema.initialize(part.permute(0,2,3,1).reshape(-1,part.shape[1]))
