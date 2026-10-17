import math
import numpy as np
from full_test_local import error_sums,aggregate,VARIANTS


def test_mask_and_pixel_weighted_aggregation():
    target=np.zeros((2,1,1,2),dtype=np.float32)
    prediction=np.array([[[[3,np.nan]]],[[[4,0]]]],dtype=np.float32)
    mask=np.array([[[[True,False]]],[[[True,True]]]])
    values=error_sums(prediction,target,mask)
    assert values==[(1,9.,3.),(2,16.,4.)]
    parts=[{'records':1,'sums':{v:dict(pixels=n,sse=sse,sae=sae) for v in (*VARIANTS,'bicubic')}} for n,sse,sae in values]
    result=aggregate(parts)['x0']
    assert result['records']==2 and result['valid_pixels']==3
    assert result['rmse_c']==math.sqrt(25/3) and result['mae_c']==7/3


def test_test_chunk_and_microbatch_tail():
    total,chunk,batch=141524,512,16
    ranges=[(a,min(a+chunk,total)) for a in range(0,total,chunk)]
    assert ranges[-1]==(141312,141524)
    assert sum(b-a for a,b in ranges)==total
    samples=[i for a,b in ranges for i in range(a,b)]
    assert samples==list(range(total))
    assert (ranges[-1][1]-ranges[-1][0])%batch==4
