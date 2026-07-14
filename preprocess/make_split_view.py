"""
make_split_view.py — create a symlink-only view of the train/test split so you
can eyeball which photos went where.

Symlinks take ~no disk space (they point at the original images; nothing is
copied). This is purely for human inspection — the dataset loader does NOT use
these folders, it reads splits.json directly.

Output:
    iter_campus/processed/split_view/
        train/<seq>/<ts>.jpg   -> original front image (symlink)
        test/<seq>/<ts>.jpg    -> original front image (symlink)

Run:  python make_split_view.py        (rebuilds from splits.json)
"""
import json
import os
import shutil

import itri_common as C

SPLITS = os.path.join(C.PROC_DIR, 'splits.json')
VIEW   = os.path.join(C.PROC_DIR, 'split_view')


def main():
    if not os.path.isfile(SPLITS):
        raise SystemExit(f'{SPLITS} not found — run select_test_region.py / '
                         f'build_split.py first.')
    data = json.load(open(SPLITS))

    if os.path.exists(VIEW):
        shutil.rmtree(VIEW)          # only contains symlinks — safe to rebuild

    n = {'train': 0, 'test': 0}
    for seq, lists in data['sequences'].items():
        for split in ('train', 'test'):
            d = os.path.join(VIEW, split, seq)
            os.makedirs(d, exist_ok=True)
            for ts in lists[split]:
                src = C.front_image_path(seq, ts)
                link = os.path.join(d, f'{ts}.jpg')
                if os.path.exists(src) and not os.path.lexists(link):
                    os.symlink(src, link)
                    n[split] += 1
        print(f'  {seq[:34]:34s} train {len(lists["train"]):5d} | '
              f'test {len(lists["test"]):5d}')
    print(f'[split_view] linked train={n["train"]} test={n["test"]}')
    print(f'[split_view] browse: {VIEW}/test/<seq>/  and  {VIEW}/train/<seq>/')


if __name__ == '__main__':
    main()
