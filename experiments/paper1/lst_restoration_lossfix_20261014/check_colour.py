"""Explicit CPU-only analytical representation experiment, no models or datasets."""
import argparse
import json
from pathlib import Path


def run(output):
    import torch
    from .representations import Representation
    from .config import file_digest, write_json
    from .runtime import source_hashes
    torch.set_num_threads(2)
    evidence = Path('runs/paper1/lst_residual_scale_20260918/residual_scale.json')
    scale = json.loads(evidence.read_text(encoding='utf-8'))
    if scale['fit_split'] != 'train' or scale['samples'] != 218975:
        raise ValueError('Expected frozen HLP v2 training-only scale')
    tau = scale['residual_std_celsius']
    report = {'scope': 'synthetic scalar mapping only; not model reconstruction or HLP scores',
              'device': 'cpu', 'models_initialized': 0, 'training_jobs_started': 0,
              'data_records_read': 0, 'A_celsius': 120., 'tau_celsius': tau,
              'scale_evidence': str(evidence), 'scale_sha256': file_digest(evidence),
              'source_hashes': source_hashes(), 'results': {}}
    def stats(error):
        return {'rmse_celsius': float(error.square().mean().sqrt()),
                'max_abs_celsius': float(error.abs().max())}
    for kind in ('G', 'C', 'C_power', 'C_asinh'):
        rep = Representation(dict(kind=kind, A_celsius=120., tau_celsius=tau))
        row = {}
        for support in (10., 120.):
            field = torch.linspace(-support, support, 100001).reshape(1, 1, 1, -1)
            encoded = rep.encode(field)
            error = rep.decode(encoded) - field
            row[f'uniform_minus_plus_{support:g}C'] = {'float32_roundtrip': stats(error)}
            if error.abs().max() > .05:
                raise AssertionError(f'{kind}: FP32 roundtrip error above declared 0.05 C technical tolerance')
            # Diagnostic ONLY: production mappings retain floating point values.
            rgb = (encoded + 1.) * .5
            rgb8 = rgb.clamp(0., 1.).mul(255).round().div(255)
            row[f'uniform_minus_plus_{support:g}C']['hypothetical_8bit_rgb_roundtrip'] = stats(rep.decode(2.*rgb8-1.) - field)
        x = torch.tensor([[[[-8., -1., 0., 1., 8.]]]], requires_grad=True)
        encoded = rep.encode(x)
        measured = sum(torch.autograd.grad(encoded[:, j].sum(), x, retain_graph=True)[0].square()
                       for j in range(rep.channels))
        predicted = rep.local_jacobian_squared(x)
        torch.testing.assert_close(measured, predicted, rtol=2e-5, atol=1e-7)
        decoded = rep.decode(encoded)
        grad = torch.autograd.grad(decoded.sum(), x)[0]
        if not torch.isfinite(grad).all():
            raise AssertionError('Nonfinite inverse gradient')
        torch.testing.assert_close(grad, torch.ones_like(grad), rtol=5e-4, atol=5e-4)
        row['local_jacobian_checked'] = True
        row['roundtrip_gradient_max_error'] = float((grad-1).abs().max())
        # No hidden clipping for supported colour input domain.
        if kind != 'G':
            try: rep.encode(torch.full((1,1,1,1),121.))
            except ValueError: row['out_of_support_rejected'] = True
            else: raise AssertionError('Out of support input silently accepted')
        report['results'][kind] = row
    report['cuda_context_initialized'] = torch.cuda.is_initialized()
    if report['cuda_context_initialized']:
        raise AssertionError('This check must not initialize CUDA')
    report['status'] = 'analytical_checks_passed_not_model_preflight'
    write_json(Path(output)/'mapping_checks.json',report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',required=True)
    result = run(p.parse_args().output)
    print(json.dumps({'status':result['status'],'results':result['results']},ensure_ascii=False,indent=2))
