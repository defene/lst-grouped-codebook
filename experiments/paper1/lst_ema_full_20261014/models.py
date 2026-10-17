import torch
from torch.utils.data import default_collate
from experiments.paper1.lst_restoration_lossfix_20261014.models import build_model as original_model
from experiments.paper1.lst_ema_pilot_20261014.quantizers import EMAVQ, EMAGrouped, controllers, initialize_from_latents

def build_model(config):
    m=original_model(config)
    if config['ema_method']!='baseline':
        wrapper=EMAVQ if config['model']['kind']=='vq' else EMAGrouped
        m.quantizer=wrapper(m.quantizer,config['ema_method']=='ema_maint')
    return m

def initialize(model,dataset,config,representation,device):
    if config['ema_method']!='ema_maint':return
    from .data import EpochSampler
    from .objective import model_inputs
    from .runtime import amp,move_batch
    from .epoch_validation import isolated_inference
    import itertools
    order=list(itertools.islice(EpochSampler(len(dataset),config['train']['sample_order_seed'],0),config['train']['effective_batch']))
    latents=[]
    with isolated_inference(model,device,config['seed']):
        for start in range(0,len(order),config['train']['micro_batch']):
            b=move_batch(default_collate([dataset[i] for i in order[start:start+config['train']['micro_batch']]]),device)
            inputs=model_inputs(b,representation);inputs.pop('patch_missing')
            with amp(config,device):latents.append(model._encode(model.input_adapter(**inputs)).float())
        initialize_from_latents(model.quantizer,torch.cat(latents))
