"""Tiny CPU autograd probes of the frozen mechanisms; no optimizer or training."""
import json
from pathlib import Path
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/torch_models_20261013/sources/stage2_vq_20261005_attempt02/experiments/paper1/lst_restoration'
package=types.ModuleType('frozen_loss_probe')
package.__path__=[str(SOURCE)]
package.VERSION='lst-restoration-v4-colour-20260927'
sys.modules['frozen_loss_probe']=package
from frozen_loss_probe.quantizers import VectorQuantizer
from frozen_loss_probe.stage2_models import DriftingVectorQuantizer
from frozen_loss_probe.representations import Representation


def run():
    torch.set_num_threads(2)
    results={}
    for kind in ['gradient_vq','drifting_vq']:
        torch.manual_seed(17)
        quantizer=(VectorQuantizer(4,16) if kind=='gradient_vq' else DriftingVectorQuantizer(
            dict(latent_dim=4,codebook_size=16,drift_tau=.2,drift_beta=.25,drift_tokens=32)))
        quantizer.train()
        latent=torch.randn(2,4,4,4,requires_grad=True)
        output=quantizer(latent)
        reconstruction=output['quantized'].square().mean()
        gradient=torch.autograd.grad(reconstruction,latent,allow_unused=True,retain_graph=True)[0]
        energy_gradient=torch.autograd.grad(output['loss'],latent,allow_unused=True)[0]
        results[kind]={'reconstruction_to_encoder_latent_grad_exists':gradient is not None,
                       'reconstruction_to_encoder_latent_grad_norm':float(gradient.norm()) if gradient is not None else 0.,
                       'bottleneck_to_encoder_latent_grad_norm':float(energy_gradient.norm()) if energy_gradient is not None else 0.}
    reps={k:Representation({'kind':k,'A_celsius':120.,'inverse_gradient_policy':'saturated_softpath_v1'}) for k in ['C','C_affine']}
    torch.manual_seed(63)
    centers=torch.cat([torch.zeros(1,3,1,1),torch.rand(20000,3,1,1)*2-1])
    delta=torch.randn_like(centers)*1e-4
    decoded=reps['C'].decode(centers)
    shifted=reps['C'].decode(centers+delta)
    difference=(decoded-shifted).abs().flatten()
    i=int(difference.argmax())
    results['C_inverse_discontinuity_example']={
        'encoded_before':centers[i].flatten().tolist(),'encoded_after':(centers+delta)[i].flatten().tolist(),
        'encoded_l2_delta':float(delta[i].norm()),'C_temperature_before':float(decoded[i].item()),
        'C_temperature_after':float(shifted[i].item()),'C_temperature_jump':float(difference[i]),
        'C_affine_temperature_jump':float((reps['C_affine'].decode(centers[i:i+1])-reps['C_affine'].decode((centers+delta)[i:i+1])).abs().item()),
        'scope':'Synthetic existence proof of inverse discontinuity; not attribution of every observed training spike'}
    results['scope']='CPU-only toy tensors; frozen source; no optimizer steps, no actual model fitting'
    out=ROOT/'outputs/loss_diagnostics_20261014/mechanism_probes.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
