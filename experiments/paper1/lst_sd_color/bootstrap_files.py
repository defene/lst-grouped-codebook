"""One-time isolated trainer adaptation; never modify A05/A06 frozen sources."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE.parent/'lst_residual_scale'
(HERE/'data.py').write_bytes((old/'data.py').read_bytes())
s=(old/'train.py').read_text(encoding='utf-8')
a=s.index('def source_hashes():')
b=s.index('\n\ndef utc():',a)
s=s[:a]+'def source_hashes():\n    return runtime.source_hashes()\n'+s[b:]
s=s.replace("'pretrained_weights': False", "'pretrained_weights': c['pretrained']")
s=s.replace("for directory in (HERE, Path(runtime.__file__).parent, Path(engine.__file__).parent, Path(runtime.base.__file__).parent):", "for directory in (HERE, Path(engine.__file__).parent, HERE.parents[2]/'eo_data'):")
s=s.replace("with engine.amp_context(c, device):\n                pred = model", "with torch.autocast('cuda',enabled=c['eval']['amp'],dtype=torch.bfloat16):\n                pred = model")
s=s.replace('    observed = []','    observed = []\n    diagnostics = {}')
s=s.replace("            metrics.add('lst', pred['lst'], moved['lst'], batch['sample_id'], batch['tile_id'])", """            metrics.add('lst', pred['lst'], moved['lst'], batch['sample_id'], batch['tile_id'])
            for view in pred['lst']:
                for key in ('kl','posterior_std','off_path_mse','saturated_fraction'):
                    diagnostics[key]=diagnostics.get(key,0.)+float(view[key])*len(batch['sample_id'])/len(pred['lst'])""")
s=s.replace("        result.update(split=split", "        result['vae_diagnostics']={k:v/total for k,v in diagnostics.items()}\n        result.update(split=split")
s=s.replace("stopped = (output/'STOP').exists() or", "stopped = (Path(c['experiment']['output_root'])/'STOP').exists() or (output/'STOP').exists() or")
s=s.replace("    ds = DatasetV2(c, 'train')\n    state = read_checkpoint", "    ds = DatasetV2(c, 'train')\n    state = read_checkpoint")
(HERE/'train.py').write_text(s,encoding='utf-8')
