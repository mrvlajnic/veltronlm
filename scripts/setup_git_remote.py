"""Create a private GitHub repository using the credential already stored in the
Windows Credential Manager, then wire it up as `origin`.

The token is never printed. It is read through `git credential fill`, used for the
GitHub API call, and discarded.

    python scripts/setup_git_remote.py --name veltronlm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"


def git_credential_fill(host: str = "github.com") -> tuple[str, str] | None:
    """Ask git's configured credential helper for the stored username/password."""
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input=f"protocol=https\nhost={host}\n\n",
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return None
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    if out.get("password"):
        return out.get("username", ""), out["password"]
    return None


def api(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "VeltronLM-setup",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:400]}
    except Exception as e:
        return -1, {"error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="veltronlm")
    ap.add_argument("--description", default="A real 4B-parameter open-weight "
                                            "decoder-only Transformer and "
                                            "customer-support AI system")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()

    cred = git_credential_fill()
    if cred is None:
        print("No GitHub credential available in the Windows Credential Manager.", file=sys.stderr)
        print("Run `git push` once interactively to store one, or set GITHUB_TOKEN.", file=sys.stderr)
        return 2
    user, token = cred
    print(f"credential found for user: {user or '(unknown)'}")

    status, who = api("GET", "/user", token)
    if status != 200:
        print(f"authentication failed: {status} {json.dumps(who)[:300]}", file=sys.stderr)
        return 2
    login = who.get("login", user)
    print(f"authenticated as: {login}")

    status, repo = api("GET", f"/repos/{login}/{args.name}", token)
    if status == 200:
        print(f"repository already exists: {repo['html_url']}")
        clone_url = repo["clone_url"]
    elif status == 404:
        status, repo = api("POST", "/user/repos", token, {
            "name": args.name,
            "description": args.description,
            "private": args.private,
            "auto_init": False,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
        })
        if status not in (200, 201):
            print(f"could not create repository: {status} {json.dumps(repo)[:400]}", file=sys.stderr)
            return 1
        print(f"created {'private' if args.private else 'public'} repository: {repo['html_url']}")
        clone_url = repo["clone_url"]
    else:
        print(f"unexpected API response: {status} {json.dumps(repo)[:300]}", file=sys.stderr)
        return 1

    subprocess.run(["git", "remote", "add", "origin", clone_url], check=False)
    remotes = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print("\nremotes:")
    print(remotes.stdout.strip())

    print(f"\npush with:\n  git push -u origin main")
    print(f"clone URL: {clone_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())