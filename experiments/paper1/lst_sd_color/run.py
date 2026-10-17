import argparse
from pathlib import Path
import torch
from eo_denoise import engine
from eo_denoise.config import fingerprint
from .prepare import read
from . import runtime,train


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['train','evaluate','pretrained'])
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--stop-after',type=int)
    a=p.parse_args();c=read(a.config);out=Path(a.output)
    if a.stage=='train':train.train(c,out,a.stop_after)
    elif a.stage=='evaluate':train.final_evaluate(c,out)
    else:
        engine.seed_everything(c['seed']);torch.set_num_threads(4)
        device=engine.device_for(c)
        model=runtime.ExperimentModel(c).to(device).eval()
        digest=c['pretrained']['sha256']['diffusion_pytorch_model.safetensors']
        for split in ('val','test'):
            for name,coverage in [('clean',0.),('noisy10',.1)]:
                dest=out/f'final_{split}_{name}'
                if (dest/'metrics.json').exists():
                    m=read(dest/'metrics.json')
                    assert m['checkpoint_sha256']==digest and m['config_sha256']==fingerprint(c)
                    continue
                train.evaluate(model,c,device,split,coverage,dest,digest)


if __name__=='__main__':main()
