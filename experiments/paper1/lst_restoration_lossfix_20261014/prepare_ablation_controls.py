"""Prepare four disabled, freshly identified controls, never start jobs."""
import copy
import json
from pathlib import Path
import uuid
from .config import validate, write_json
from .runtime import source_hashes


def prepare(base='runs/paper1/lst_restoration_lossfix_20261014'):
    root=Path(base)
    destination=root/'controls'
    if destination.exists():
        raise ValueError('Use a new control destination; do not overwrite frozen identities')
    pairs=[]
    for name,section,key,value in (
        ('sd_C','representation','colour_inverse','nearest_segment'),
        ('vq_G','model','vq_geometry','euclidean'),
        ('vq_drifting_G','model','drift_reconstruction_gradient','codes_only'),
        ('grouped_g8k256_C_power','representation','colour_inverse','nearest_segment'),
    ):
        c=copy.deepcopy(json.loads((root/'configs'/f'{name}.json').read_text(encoding='utf-8')))
        c[section][key]=value
        c['name'] += '_mechanism_control'
        c['tracking'].update(run_id=uuid.uuid4().hex[:12],name=c['name'],group='lossfix_controls_20261014')
        assert validate(c)['ok']
        write_json(destination/f'{name}_control.json',c)
        pairs.append(dict(model=name,candidate=f'configs/{name}.json',control=f'controls/{name}_control.json',
                          changed_field=f'{section}.{key}',control_value=value))
    write_json(root/'CONTROLLED_CHECK_PLAN.json',dict(status='prepared_not_launched',
        scope='Four paired mechanism controls; these configs retain formal five-epoch budgets.',
        technical_pilot='A shorter real-data pilot needs a separately budgeted runner; not executed or represented as formal training.',
        pairs=pairs,source_hashes=source_hashes(),
        acceptance=['Compare reconstruction and full validation, not total loss alone.',
                    'Inspect code usage, per-term latent gradient norms and global clipping norm.',
                    'Verify decode/index replay and new memory profile before any formal launch.'],
        input_data='HLP v2 train only for fitting, full val for selection, no test selection',
        all_execution_disabled=True))
    print('Prepared four disabled controls; no experiments submitted')


if __name__=='__main__':
    prepare()
