"""Stage an identical Transformers source tree on the allocation's local disk.

Transformers recursively reads its Python sources to build its lazy import index,
even when Python bytecode is warm. Keep the installed environment authoritative;
only a verified, same-cluster snapshot may supply those reads locally.
"""

import json
import os
import pathlib
import sys


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def _write_json(path, data):
    temporary = path.with_name(path.name + ".partial-" + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def build(root, work):
    """Run with the target environment's Python, before validating that environment."""
    import subprocess
    import sysconfig
    import tarfile

    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    source = site / "transformers"
    environment = pathlib.Path(sys.prefix).resolve()
    if source.is_symlink() or environment not in source.resolve().parents:
        raise ValueError("Source cache requires Transformers installed inside this environment")
    records = list(site.glob("transformers-*.dist-info/RECORD"))
    if len(records) != 1:
        raise ValueError("Expected one installed Transformers distribution")
    record = records[0]
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if not cluster:
        raise ValueError("Build a source cache in its Slurm cluster")
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / (".transformers-source-" + str(os.getpid()) + ".tar.gz")
    files = {}

    def include(member):
        path = pathlib.PurePosixPath(member.name)
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            return None
        if not (member.isfile() or member.isdir()):
            raise ValueError("Source tree contains a link or special file")
        if member.isfile():
            original = site / path
            before = original.stat()
            files[member.name] = {"size": before.st_size, "sha256": _sha256(original)}
            after = original.stat()
            if (after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError("Installed source changed while making the snapshot")
        return member

    try:
        with tarfile.open(str(temporary), "w:gz") as bundle:
            bundle.add(str(source), arcname="transformers", filter=include)
        if "transformers/__init__.py" not in files:
            raise ValueError("Missing Transformers package initializer")
        digest = _sha256(temporary)
        archive = cache / ("transformers-source-" + digest + ".tar.gz")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "cluster": cluster, "python_env": str(environment),
        "python_cache_tag": sys.implementation.cache_tag,
        "git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                            universal_newlines=True).strip(),
        "archive": str(archive), "sha256": digest,
        "record": str(record), "record_sha256": _sha256(record), "files": files,
    }
    _write_json(cache / "transformers-source-candidate.json", result)
    return result


def stage(root, work, local, candidate=False):
    """Validate every member and byte hash before publishing a node-local package."""
    import subprocess
    import tarfile
    import tempfile

    pointer = work / "cache" / ("transformers-source-candidate.json" if candidate
                               else "transformers-source-ready.json")
    meta = json.loads(pointer.read_text())
    environment = (work / "envs/verl").resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   universal_newlines=True).strip()
    if (not os.environ.get("SLURM_CLUSTER_NAME")
            or meta.get("cluster") != os.environ["SLURM_CLUSTER_NAME"]
            or meta.get("git_head") != head or meta.get("python_env") != str(environment)):
        return {"source_cache": "skipped", "reason": "cluster, environment or checkout differs"}
    archive = pathlib.Path(meta["archive"]).resolve()
    record = pathlib.Path(meta["record"]).resolve()
    if ((work / "cache").resolve() not in archive.parents or environment not in record.parents
            or record.name != "RECORD"):
        raise ValueError("Source cache path is outside this work directory")
    if _sha256(archive) != meta["sha256"] or _sha256(record) != meta["record_sha256"]:
        raise ValueError("Source archive or installed distribution checksum differs")
    local.mkdir(parents=True, exist_ok=True)
    target = local / "python-packages"
    if target.exists() or target.is_symlink():
        if (not target.is_symlink()
                and json.loads((target / ".ready.json").read_text()) == meta):
            return {"source_cache": "loaded", "sha256": meta["sha256"], "already_staged": True}
        raise ValueError("Refusing to replace an existing local package directory")
    with tempfile.TemporaryDirectory(prefix=".python-packages-", dir=str(local)) as name:
        temporary = pathlib.Path(name)
        with tarfile.open(str(archive), "r:gz") as bundle:
            members = bundle.getmembers()
            seen = set()
            for member in members:
                path = pathlib.PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or not path.parts
                        or path.parts[0] != "transformers" or member.name in seen
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Unsafe source archive member")
                seen.add(member.name)
                if member.isfile() and (member.name not in meta["files"]
                                        or member.size != meta["files"][member.name]["size"]):
                    raise ValueError("Unexpected source archive file")
            bundle.extractall(str(temporary), members=members)
        actual = {str(p.relative_to(temporary)) for p in temporary.rglob("*") if p.is_file()}
        if actual != set(meta["files"]):
            raise ValueError("Source archive file inventory differs")
        for relative, entry in meta["files"].items():
            if _sha256(temporary / relative) != entry["sha256"]:
                raise ValueError("Extracted source checksum differs")
        _write_json(temporary / ".ready.json", meta)
        # Rename the contents' parent atomically; TemporaryDirectory then has nothing to remove.
        os.rename(str(temporary), str(target))
    return {"source_cache": "loaded", "sha256": meta["sha256"], "files": len(actual),
            "bytes": sum(entry["size"] for entry in meta["files"].values())}


def activate():
    """Called before any Transformers import, including in Ray's child interpreters."""
    local = os.environ.get("LOCAL_DIR")
    if not local:
        return False
    target = pathlib.Path(local) / "python-packages"
    marker = target / ".ready.json"
    try:
        if (target.is_symlink() or marker.is_symlink() or (target / "transformers").is_symlink()
                or not marker.is_file() or not (target / "transformers/__init__.py").is_file()):
            return False
        meta = json.loads(marker.read_text())
        if (not isinstance(meta, dict)
                or meta.get("python_env") != str(pathlib.Path(sys.prefix).resolve())
                or meta.get("python_cache_tag") != sys.implementation.cache_tag
                or not os.environ.get("SLURM_CLUSTER_NAME")
                or meta.get("cluster") != os.environ["SLURM_CLUSTER_NAME"]):
            return False
    except (OSError, ValueError, TypeError):
        return False
    path = str(target)
    if path not in sys.path:
        sys.path.insert(0, path)
    return True


if __name__ == "__main__":
    mode, root, work, *rest = sys.argv[1:]
    if mode == "build":
        result = build(pathlib.Path(root), pathlib.Path(work))
        print(json.dumps({k: v for k, v in result.items() if k != "files"}))
    elif mode in ("stage", "stage-candidate"):
        result = stage(pathlib.Path(root), pathlib.Path(work), pathlib.Path(rest[0]),
                       candidate=mode == "stage-candidate")
        print(json.dumps(dict(result, node=os.uname().nodename)), flush=True)
    else:
        raise ValueError("Expected build, stage or stage-candidate")
