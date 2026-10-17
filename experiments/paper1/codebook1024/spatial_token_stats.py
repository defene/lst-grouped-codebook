"""Categorical token statistics and masked, sample-equal spatial decomposition."""
import numpy as np


def components(values, support):
    """Per-image channel means and mean centered variance; no mixing channel means."""
    values=np.asarray(values,dtype=np.float64)
    support=np.asarray(support,dtype=bool).reshape(len(values),-1)
    values=values.reshape(len(values),values.shape[1],-1)
    counts=support.sum(1)
    if np.any(counts==0):raise ValueError('Empty spatial support')
    means=(values*support[:,None]).sum(2)/counts[:,None]
    residual=values-means[:,:,None]
    within=(residual**2*support[:,None]).sum((1,2))/(counts*values.shape[1])
    return means,within


def variance_summary(means,within):
    between=float(np.var(means,axis=0).mean())
    intra=float(np.mean(within))
    return dict(between=between,within=intra,total=between+intra,
                between_fraction=between/(between+intra) if between+intra>0 else None)


def reconstruction_parts(pred,target,support):
    pm,pw=components(pred,support);tm,tw=components(target,support)
    em,ew=components(np.asarray(pred)-np.asarray(target),support)
    mean_mse=np.mean(em**2,axis=1)
    # Exact orthogonal decomposition of MSE into image mean and image-centered errors.
    return dict(pred_mean=pm,pred_within=pw,target_mean=tm,target_within=tw,
                mean_mse=mean_mse,centered_mse=ew,mse=mean_mse+ew)


def entropy_rows(counts):
    counts=np.asarray(counts,dtype=np.float64)
    n=counts.sum(1)
    if np.any(n==0):raise ValueError('Empty token support')
    p=counts/n[:,None]
    h=-(p*np.log2(np.where(p>0,p,1))).sum(1)
    return p,h


def token_summary(ids,support,vocabulary,permutations=0,seed=20260914):
    ids=np.asarray(ids,dtype=np.int64)
    n,side,_=ids.shape
    support=np.asarray(support,dtype=bool).reshape(n,-1)
    flat=ids.reshape(n,-1)
    keep=support.sum(1)>0
    flat,support=flat[keep],support[keep]
    n=len(flat)
    groups=np.broadcast_to(np.arange(n)[:,None],flat.shape)[support]
    values=flat[support]
    def calculate(tokens):
        counts=np.bincount(groups*vocabulary+tokens,minlength=n*vocabulary).reshape(n,vocabulary)
        p,h=entropy_rows(counts)
        marginal=p.mean(0)
        hm=float(-(marginal*np.log2(np.where(marginal>0,marginal,1))).sum())
        hw=float(h.mean())
        return counts,p,h,hm,hw
    counts,p,h,hm,hw=calculate(values)
    cardinality=(counts>0).sum(1)
    dominant=p.max(1)
    count=counts.sum(1)
    eligible=count>1
    within_random=((counts*(counts-1)).sum(1)[eligible]/(count*(count-1))[eligible]).mean() if eligible.any() else None
    between_random=float(((p.sum(0)**2).sum()-(p**2).sum())/(n*(n-1))) if n>1 else None
    spatial=[]
    for array,mask in zip(flat.reshape(n,side,side),support.reshape(n,side,side)):
        equal=[]
        for a,b,ok in [(array[:,:-1],array[:,1:],mask[:,:-1]&mask[:,1:]),
                       (array[:-1],array[1:],mask[:-1]&mask[1:])]:
            if ok.any():equal.append((a[ok]==b[ok]))
        spatial.append(float(np.concatenate(equal).mean()) if equal else np.nan)
    null=[]
    rng=np.random.default_rng(seed)
    for _ in range(permutations):
        *_,hp,hwp=calculate(rng.permutation(values))
        null.append(hp-hwp)
    mi=max(0.,hm-hw)
    return dict(images=n,scale=side,positions=int(len(values)),vocabulary=vocabulary,
        marginal_entropy_bits=hm,conditional_within_entropy_bits=hw,image_identity_mutual_information_bits=mi,
        image_identity_fraction_of_token_entropy=mi/hm if hm>0 else None,
        marginal_perplexity=2**hm,within_perplexity_geometric_mean=2**hw,
        per_image_unique_median=float(np.median(cardinality)),per_image_unique_q10_q90=np.quantile(cardinality,[.1,.9]).tolist(),
        per_image_dominant_fraction_median=float(np.median(dominant)),
        same_token_random_within_image=float(within_random) if within_random is not None else None,
        same_token_random_between_images=between_random,
        same_token_adjacent=float(np.nanmean(spatial)) if np.isfinite(spatial).any() else None,
        permutation_null_mi_mean=float(np.mean(null)) if null else None,
        permutation_null_mi_q025_q975=np.quantile(null,[.025,.975]).tolist() if null else None,
        permutation_corrected_mi=max(0.,mi-float(np.mean(null))) if null else None,
        permutation_count=len(null),one_by_one_not_evidence_for_spatial_collapse=side==1)
