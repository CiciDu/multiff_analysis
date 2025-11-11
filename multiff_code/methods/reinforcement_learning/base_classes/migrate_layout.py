import os
import json
import shutil
from typing import Iterable, List, Optional, Tuple


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _safe_symlink(target: str, link_path: str) -> None:
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            try:
                os.remove(link_path)
            except IsADirectoryError:
                os.rmdir(link_path)
            except Exception:
                pass
        os.symlink(target, link_path)
    except Exception:
        # Symlinks may fail on some FS; ignore
        pass


def _move_dir_contents(src: str, dst: str) -> None:
    """
    Move src into dst/basename(src) if dst doesn't exist; else move contents into dst.
    """
    if not os.path.exists(src):
        return
    _ensure_dir(dst)
    # If dst is empty, move the whole directory under a matching leaf name when possible
    if not os.listdir(dst):
        # try direct move when dst is the intended container (e.g., curr/best)
        for entry in os.listdir(src):
            s = os.path.join(src, entry)
            d = os.path.join(dst, entry)
            shutil.move(s, d)
        # remove empty src
        try:
            os.rmdir(src)
        except Exception:
            pass
        return
    # Otherwise, move files one-by-one
    for entry in os.listdir(src):
        s = os.path.join(src, entry)
        d = os.path.join(dst, entry)
        if os.path.exists(d):
            # avoid overwriting; append suffix
            base, ext = os.path.splitext(entry)
            d = os.path.join(dst, f"{base}_old{ext}")
        shutil.move(s, d)
    try:
        os.rmdir(src)
    except Exception:
        pass


def _read_manifest_env_params(dir_path: str) -> Optional[dict]:
    """
    Read checkpoint_manifest.json from dir_path; return env_params dict if present.
    """
    manifest_path = os.path.join(dir_path, 'checkpoint_manifest.json')
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'env_params' in data and isinstance(data['env_params'], dict):
                return data['env_params']
            # tolerate legacy payloads (raw env_kwargs)
            if all(isinstance(k, str) for k in data.keys()):
                return data
    except Exception:
        pass
    return None


def migrate_agent_dir(agent_root: str) -> Tuple[bool, List[str]]:
    """
    Migrate one agent directory from legacy layout to:
      - curr/best/
      - post/best/
      - ln/best_curr -> curr/best
      - ln/best_post -> post/best
      - meta/{manifest.json, env_params.json}
    Also moves legacy curriculum log to curr/log.csv when found.

    Returns:
      (changed: bool, notes: list[str])
    """
    notes: List[str] = []
    changed = False

    if not os.path.isdir(agent_root):
        return False, [f"Skip (not a directory): {agent_root}"]

    # New layout paths
    curr_dir = os.path.join(agent_root, 'curr')
    post_dir = os.path.join(agent_root, 'post')
    ln_dir = os.path.join(agent_root, 'ln')
    meta_dir = os.path.join(agent_root, 'meta')
    best_curr = os.path.join(curr_dir, 'best')
    best_post = os.path.join(post_dir, 'best')

    # Legacy paths
    legacy_curr = os.path.join(agent_root, 'best_model_in_curriculum')
    legacy_post = os.path.join(agent_root, 'best_model_postcurriculum')

    # Create base dirs
    for d in [curr_dir, post_dir, ln_dir, meta_dir, best_curr, best_post]:
        _ensure_dir(d)

    # Move legacy -> new layout
    if os.path.isdir(legacy_curr):
        _move_dir_contents(legacy_curr, best_curr)
        notes.append("Moved best_model_in_curriculum -> curr/best")
        changed = True
    if os.path.isdir(legacy_post):
        _move_dir_contents(legacy_post, best_post)
        notes.append("Moved best_model_postcurriculum -> post/best")
        changed = True

    # Curriculum log
    # Old location likely under legacy_curr/curriculum_log.csv
    old_log = os.path.join(best_curr, 'curriculum_log.csv')
    if not os.path.exists(old_log):
        # Try agent_root or legacy_curr direct
        alt1 = os.path.join(agent_root, 'curriculum_log.csv')
        alt2 = os.path.join(legacy_curr, 'curriculum_log.csv')
        old_log = alt1 if os.path.exists(alt1) else (
            alt2 if os.path.exists(alt2) else old_log)
    new_log = os.path.join(curr_dir, 'log.csv')
    if os.path.exists(old_log):
        _ensure_dir(curr_dir)
        try:
            shutil.move(old_log, new_log)
            notes.append("Moved curriculum_log.csv -> curr/log.csv")
            changed = True
        except Exception:
            pass

    # Symlinks
    _safe_symlink(best_curr, os.path.join(ln_dir, 'best_curr'))
    _safe_symlink(best_post, os.path.join(ln_dir, 'best_post'))

    # Meta files
    env_params = None
    # Prefer post best, then curr best, then agent_root
    for probe in [best_post, best_curr, agent_root]:
        env_params = _read_manifest_env_params(probe)
        if isinstance(env_params, dict):
            break
    if isinstance(env_params, dict):
        try:
            with open(os.path.join(meta_dir, 'env_params.json'), 'w') as f:
                json.dump(env_params, f, indent=2, default=str)
            changed = True
            notes.append("Wrote meta/env_params.json")
        except Exception:
            pass

    # Write a lightweight manifest pointing to symlinks
    try:
        manifest = {
            'algo': None,  # unknown at migration time
            'version': '1',
            'role': 'root',
            'parent': None,
            'env_params': 'meta/env_params.json' if os.path.exists(os.path.join(meta_dir, 'env_params.json')) else None,
            'agent_params': 'meta/agent_params.json' if os.path.exists(os.path.join(meta_dir, 'agent_params.json')) else None,
            'artifacts': {
                'best_curr': 'ln/best_curr',
                'best_post': 'ln/best_post'
            }
        }
        with open(os.path.join(meta_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        notes.append("Wrote meta/manifest.json")
        changed = True
    except Exception:
        pass

    return changed, notes


def find_agent_roots(root: str) -> List[str]:
    """
    Heuristically find agent directories: leaf directories that contain either
    a best model (legacy or new) or a checkpoint_manifest.json.
    """
    candidates: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip deep nesting to keep scan fast
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if depth > 6:
            continue
        # Signals of agent roots
        signals = [
            'best_model_postcurriculum', 'best_model_in_curriculum',
            os.path.join('post', 'best'), os.path.join('curr', 'best'),
            'checkpoint_manifest.json', 'best_model.zip'
        ]
        if any(s in filenames for s in ['checkpoint_manifest.json', 'best_model.zip']):
            candidates.append(dirpath)
            continue
        # Check subdirs
        if any(s in dirnames for s in ['best_model_postcurriculum', 'best_model_in_curriculum', 'post', 'curr']):
            candidates.append(dirpath)
    # Deduplicate and prefer shallower paths
    candidates = sorted(set(candidates), key=lambda p: (p.count(os.sep), p))
    return candidates


def migrate_all(root: str) -> None:
    """
    Migrate all agent directories under root.
    """
    print(f"[migrate] scanning: {root}")
    agents = find_agent_roots(root)
    print(f"[migrate] found {len(agents)} candidate agent directories")
    migrated = 0
    for a in agents:
        changed, notes = migrate_agent_dir(a)
        if changed:
            migrated += 1
            print(f"[migrate] {a}")
            for n in notes:
                print(f"  - {n}")
    print(f"[migrate] done. migrated {migrated} / {len(agents)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Migrate legacy RL agent folders to standardized layout (curr/post/ln/meta).")
    parser.add_argument(
        "root", type=str, help="Root directory to scan (e.g., multiff_analysis/RL_models)")
    args = parser.parse_args()
    migrate_all(os.path.abspath(os.path.expanduser(args.root)))
