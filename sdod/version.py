version = '0.1.0.dev0'
repo = 'unknown'
commit = 'unknown'
has_repo = False

try:
    import git
    from pathlib import Path

    try:
        r = git.Repo(Path(__file__).parents[1])
        has_repo = True

        if not r.remotes:
            repo = 'local'
        else:
            repo = r.remotes.origin.url

        commit = r.head.commit.hexsha
        status = []
        if r.is_dirty():
            status.append('dirty')
        if r.untracked_files:
            status.append(f'+{len(r.untracked_files)} untracked')
        if status:
            commit += f' ({",".join(status)})'
    except git.InvalidGitRepositoryError:
        raise ImportError()
except ImportError:
    pass

try:
    from . import _dist_info as _info
    assert not has_repo, '_dist_info should not exist when repo is in place'
    assert version == _info.version
    repo = _info.repo
    commit = _info.commit
except (ImportError, SystemError):
    pass


def info():
    g = globals()
    return { k: g[k] for k in __all__ }


__all__ = ['version', 'repo', 'commit', 'has_repo']
