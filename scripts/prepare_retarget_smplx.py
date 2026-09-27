#!/usr/bin/env python3
"""Stage supported InterMimic [T,591] SMPLX clips for ULTRA G1 retargeting."""
import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--asset-scale', choices=('080_080_080', '100_100_100'), default='080_080_080')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if source_dir == output_dir:
        raise ValueError('input and output directories must differ')
    assets = repo / 'ultra/data/assets/objects/diverse'
    pattern = re.compile(r'^(sub[0-9]+)_([a-z0-9]+)_([0-9]+)$')
    targets = {}
    for source in sorted(source_dir.glob('*.pt')):
        match = pattern.fullmatch(source.stem)
        if not match:
            continue
        object_name = match.group(2)
        asset_name = f'{object_name}_{args.asset_scale}'
        asset_dir = assets / asset_name
        if not (asset_dir / f'{asset_name}.urdf').is_file() or not (asset_dir / f'{asset_name}.obj').is_file():
            continue
        targets[f'{source.stem}_{args.asset_scale}.pt'] = source
    if not targets:
        raise RuntimeError('No supported OMOMO motions found')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in targets.items():
        link = output_dir / name
        if link.is_symlink() and link.resolve() == source:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)
    print(f'Prepared {len(targets)} motions in {output_dir}', flush=True)


if __name__ == '__main__':
    main()
