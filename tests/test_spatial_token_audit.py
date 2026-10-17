import numpy as np
from experiments.paper1.codebook1024.spatial_token_stats import components,variance_summary,reconstruction_parts,token_summary


def test_only_image_identity_has_no_within_entropy():
    ids=np.broadcast_to(np.arange(4)[:,None,None],(4,4,4))
    s=token_summary(ids,np.ones_like(ids,bool),8)
    assert s['conditional_within_entropy_bits']==0
    assert s['marginal_entropy_bits']==2
    assert s['same_token_random_within_image']==1
    assert s['same_token_random_between_images']==0


def test_shared_spatial_diversity_is_not_image_identity():
    ids=np.tile(np.array([[0,1],[0,1]]),(4,1,1))
    s=token_summary(ids,np.ones_like(ids,bool),8)
    assert s['conditional_within_entropy_bits']==1
    assert s['image_identity_mutual_information_bits']==0
    renamed=np.where(ids==0,7,3)
    t=token_summary(renamed,np.ones_like(ids,bool),8)
    assert s==t


def test_mean_shift_and_spatial_error_separate_with_masks():
    target=np.arange(24,dtype=float).reshape(2,3,2,2)
    mask=np.array([[[1,1],[1,0]],[[1,1],[1,1]]],bool)
    p=reconstruction_parts(target+2,target,mask)
    np.testing.assert_allclose(p['mean_mse'],4)
    np.testing.assert_allclose(p['centered_mse'],0)
    means,within=components(target,mask)
    summary=variance_summary(means,within)
    global_mean=means.mean(0)
    expected=np.mean([((x[:,ok]-global_mean[:,None])**2).mean() for x,ok in zip(target,mask)])
    assert np.isclose(summary['total'],expected)


def test_token_support_excludes_invalid_codes_and_null_handles_small_grids():
    ids=np.array([[[0,7],[0,7]],[[1,7],[1,7]]])
    mask=np.tile([[True,False],[True,False]],(2,1,1))
    s=token_summary(ids,mask,8,permutations=8)
    assert s['positions']==4 and s['marginal_perplexity']==2
    assert s['per_image_unique_median']==1 and s['permutation_count']==8
