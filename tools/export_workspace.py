"""Export a non-overwriting code/evidence snapshot; never copy datasets or weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TREES = ("eo_data", "eo_denoise", "experiments", "examples", "tests", "docs", "tools",
         "data_inspection/hlp_split_20260914")
ROOT_FILES = ("README.md", "AGENTS.md", ".gitignore", "PAPER1_EXPERIMENT_PLAN.md",
              "ICLR_CROSS_VIEW_DISCRETE_TOKENIZATION_PLAN.md", "create_validation_split.py",
              "extract_all_data.py", "inspect_data.py", "profile_normalization.py",
              "prepare_data.ps1", "run_paper1.ps1", "setup_paper1.ps1",
              "data_inspection/FINAL_DATASET_SUMMARY.md")
EVIDENCE = (
    "output/workspace_organization_20260923/data_preflight.json",
    "runs/paper1/lst_v2_5epoch_20260915/REPORT_ZH.md",
    "runs/paper1/lst_v2_5epoch_20260915/status.json",
    "runs/paper1/lst_residual_scale_20260918/REPORT_ZH.md",
    "runs/paper1/lst_residual_scale_20260918/status.json",
    "runs/paper1/lst_sd_color_20260921/REPORT_ZH.md",
    "runs/paper1/lst_sd_color_20260921/status.json",
    "output/efdiff_hlc_epoch1_deadline_io16_20260916/completion_verified.json",
    "output/efdiff_hlc_epoch1_deadline_io16_20260916/full_test_20260921/status.json",
    "output/prithvi_partner_final_20260923/README.md",
)
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".next", "bench"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".ckpt", ".safetensors", ".zip", ".eoz"}
EXCLUSIONS = ["raw datasets", "processed_data (all versions)", "model weights",
              "full runs/output and report images", "reports websites", "virtual environments",
              "credentials/.env files", "caches, logs, benchmarks and .git history"]
MANIFEST = "WORKSPACE_MANIFEST.json"


def selected_files(root: Path):
    selected = set()
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required source missing or symlinked: {path}")
        selected.add(path)
    for name in TREES:
        base = root / name
        if not base.is_dir() or base.is_symlink():
            raise FileNotFoundError(f"Required tree missing or symlinked: {base}")
        for folder, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not d.startswith('.venv') and not (Path(folder) / d).is_symlink())
            for filename in files:
                path = Path(folder) / filename
                if path.is_symlink() or path.suffix.lower() in SKIP_SUFFIXES:
                    continue
                if filename.startswith('.env') and filename != '.env.example':
                    continue
                if filename.endswith(('-wal', '-shm')):
                    continue
                selected.add(path)
    missing = []
    for name in EVIDENCE:
        path = root / name
        if path.is_file() and not path.is_symlink():
            selected.add(path)
        else:
            missing.append(name)
    return sorted(selected, key=lambda p: p.relative_to(root).as_posix()), missing


def verify(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        manifest = json.loads(archive.read(MANIFEST))
        expected = manifest['files']
        if set(names) != set(expected) | {MANIFEST}:
            raise ValueError('ZIP inventory differs from manifest')
        for name, entry in expected.items():
            digest = hashlib.sha256()
            count = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    count += len(block)
            if count != entry['size'] or digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Content mismatch: {name}')
    return {'status': 'verified', 'archive': str(path), 'files': len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--output', type=Path, help='Create a new ZIP (never overwrite)')
    action.add_argument('--verify', type=Path, help='Verify an existing snapshot without extracting')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify.resolve()), indent=2))
        return
    root = args.root.resolve()
    files, missing = selected_files(root)
    summary = {'files': len(files), 'uncompressed_bytes': sum(p.stat().st_size for p in files),
               'by_top_directory': dict(Counter(p.relative_to(root).parts[0] for p in files)),
               'excluded': EXCLUSIONS, 'missing_optional_evidence': missing}
    if args.output:
        target = args.output.resolve()
        if any(target.is_relative_to(root / name) for name in TREES):
            raise ValueError('Export destination must be outside source trees')
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'scope': 'code and selected evidence; not a dataset or resume backup',
                    'selection': summary, 'files': {}}
        # Exclusive creation: failures leave an inspectable incomplete ZIP, never replace old exports.
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                content = path.read_bytes()
                archive.writestr(name, content)
                manifest['files'][name] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        summary.update(verify(target))
        summary['archive_bytes'] = target.stat().st_size
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
