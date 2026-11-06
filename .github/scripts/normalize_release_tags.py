"""Prepare release tags in an expendable clone only; never fetches or pushes."""

import json
import subprocess
from pathlib import Path


def git(repo, *args, data=None):
    result = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout.decode().strip()


def normalize(repo, mapping_file):
    records = json.loads(Path(mapping_file).read_text(encoding="utf-8"))
    updates = []
    for record in records:
        ref = "refs/tags/" + record["tag"]
        public_oid = record["public_object"]
        new_commit = record["rewritten_commit"]
        assert git(repo, "rev-parse", ref) == public_oid, ref
        assert git(repo, "rev-parse", ref + "^{commit}") == record["old_commit"], ref
        assert git(repo, "rev-parse", ref + "^{tree}") == record["tree"], ref
        assert git(repo, "rev-parse", new_commit + "^{tree}") == record["tree"], ref
        git(repo, "merge-base", "--is-ancestor", new_commit, "HEAD")
        kind = git(repo, "cat-file", "-t", public_oid)
        if kind == "tag":
            original = subprocess.check_output(["git", "-C", str(repo), "cat-file", "tag", public_oid])
            first, rest = original.split(b"\n", 1)
            assert first == b"object " + record["old_commit"].encode()
            # A signature on the original tag cannot authenticate its replacement.
            for signature in (b"-----BEGIN PGP SIGNATURE-----", b"-----BEGIN SSH SIGNATURE-----"):
                rest = rest.split(signature, 1)[0]
            replacement = b"object " + new_commit.encode() + b"\n" + rest
            local_oid = git(repo, "hash-object", "-t", "tag", "-w", "--stdin", data=replacement)
        else:
            assert kind == "commit", kind
            local_oid = new_commit
        updates.append((ref, public_oid, local_oid))
    transaction = "".join(f"update {ref} {new} {old}\n" for ref, old, new in updates)
    git(repo, "update-ref", "--stdin", data=transaction.encode())
    return updates


def restore(repo, updates):
    transaction = "".join(f"update {ref} {old} {new}\n" for ref, old, new in updates)
    git(repo, "update-ref", "--stdin", data=transaction.encode())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--apply-in-disposable-clone", action="store_true", required=True)
    args = parser.parse_args()
    applied = normalize(args.repo.resolve(), args.mapping.resolve())
    print(json.dumps({"local_tags_prepared": len(applied), "remote_tags_changed": False}))
