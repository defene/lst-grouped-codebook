import math
from experiments.paper1.lst_restoration_lossfix_20261014.config import COUNTS, PREPARED, PROJECT, digest, file_digest, write_json

def validate(c):
    from .epoch_validation import protocol_errors
    errors=protocol_errors(c)
    checks={
        'full HLP v2 LST required': c['data']['prepared']==PREPARED and c['data']['counts']==COUNTS and c['data']['shape']==[1,256,256],
        'registered G comparison required': c['representation']['kind']=='G' and c['model']['kind'] in ('vq','grouped_g8k256'),
        'registered EMA method required': c['ema_method'] in ('baseline','ema','ema_maint'),
        'five full epochs and effective batch64 required': c['train']['epochs']==5 and c['train']['effective_batch']==64,
        'exact full record budget required': c['train']['record_presentations']==5*COUNTS['train'],
        'exact update budget required': c['train']['updates']==5*math.ceil(COUNTS['train']/64),
        'microbatch must divide effective batch': 64%c['train']['micro_batch']==0,
        'train initialization and val-only selection required': c['train']['selection']=='best_validation_rmse' and not c['model'].get('pretrained_path'),
        'local tracking only': not c.get('tracking',{}).get('enabled',False),
        'no extra latent RMS change': c['model'].get('vq_geometry','euclidean')=='euclidean',
    }
    errors.extend(k for k,v in checks.items() if not v)
    return {'ok':not errors,'errors':errors}
