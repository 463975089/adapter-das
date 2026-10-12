#!/usr/bin/env python3
"""Self-hosted Gitee MR watcher: PR opened/updated -> L1 GPU gate -> PR comment.

Why self-hosted: the GPU tiers need real HCU hardware (gfx938 + DTK stack),
which only exists inside the nmz26 container -- Gitee Go cloud runners are
CPU-only and cover only L0 (see .workflow/l0-gate.yml).

Flow per open MR (pull request) whose head sha is untested:
  1. fetch the PR head into a DEDICATED clone (never /root/hcu_t2, which
     serves live validation services)
  2. checkout a git worktree
  3. run ci/l1_gpu_gate.sh (auto-picks a free GPU; exit 2 = no free card ->
     state stays "queued" and it retries next poll)
  4. post the verdict back as a PR comment (needs token; without one the
     watcher still tests and records state, it just cannot comment)

Secrets policy: the Gitee access token NEVER lives in this repo. It is read
from env GITEE_TOKEN or a one-line file ~/.hcu_ci_token owned by the
operator (create with:  echo -n <token> > ~/.hcu_ci_token && chmod 600 ...).

Usage:
  python3 ci/mr_watcher.py --once        # single sweep (cron/ci friendly)
  python3 ci/mr_watcher.py               # loop, poll every 90s
Requires: git, curl (or urllib only -- stdlib used), rocm-smi in PATH for
GPU picking (delegated to l1_gpu_gate.sh).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

API = "https://gitee.com/api/v5"
# repo location is deployment config, not code: always set HCU_CI_REPO
# (e.g. via env or ~/.hcu_ci_env) — survives repo migration to another host.
REPO_SLUG = os.environ.get("HCU_CI_REPO", "")
if not REPO_SLUG:
    REPO_SLUG_FILE = os.path.join(
        os.environ.get("HCU_CI_WORKDIR", os.path.expanduser("~/hcu_ci")),
        "repo_slug")
    try:
        REPO_SLUG = open(REPO_SLUG_FILE, encoding="utf-8").read().strip()
    except OSError:
        pass
if not REPO_SLUG:
    sys.exit("mr_watcher: set HCU_CI_REPO=<owner>/<repo> (or write it to"
             " ~/hcu_ci/repo_slug) — repo location is deployment config")

WORKDIR = os.environ.get("HCU_CI_WORKDIR", os.path.expanduser("~/hcu_ci"))
STATE_FILE = os.path.join(WORKDIR, "state.json")
CLONE_DIR = os.path.join(WORKDIR, "src")
WT_DIR = os.path.join(WORKDIR, "wt")


# ---------------------------------------------------------------------------
def load_token():
    tok = os.environ.get("GITEE_TOKEN", "").strip()
    if tok:
        return tok
    p = os.path.expanduser(os.environ.get("HCU_CI_TOKEN_FILE", "~/.hcu_ci_token"))
    if os.path.isfile(p):
        return open(p).read().strip()
    return ""


def api_get(path, token):
    url = f"{API}{path}"
    if token:
        sep = "&" if "?" in url else "?"
        url += f"{sep}access_token={token}"
    req = urllib.request.Request(url, headers={"User-Agent": "hcu-ci"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def api_post_comment(pr_number, body, token):
    if not token:
        return False
    url = f"{API}/repos/{REPO_SLUG}/pulls/{pr_number}/comments"
    data = json.dumps({"body": body, "access_token": token}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "hcu-ci"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status in (200, 201)
    except Exception as e:
        print(f"  comment post failed: {e}", flush=True)
        return False


def load_state():
    if os.path.isfile(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}


def save_state(st):
    os.makedirs(WORKDIR, exist_ok=True)
    json.dump(st, open(STATE_FILE, "w"), indent=1, ensure_ascii=False)


def sh(cmd, cwd=None, timeout=600):
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                          timeout=timeout, capture_output=True, text=True)


# ---------------------------------------------------------------------------
def ensure_clone(token):
    if os.path.isdir(os.path.join(CLONE_DIR, ".git")):
        return True
    os.makedirs(WORKDIR, exist_ok=True)
    url = f"https://gitee.com/{REPO_SLUG}.git"
    r = sh(["git", "clone", "--quiet", url, CLONE_DIR], timeout=300)
    if r.returncode != 0:
        # private repo: retry with the token via an ephemeral askpass (token
        # stays in process memory / env, never in any file or remote config)
        if token:
            helper = ("!f() { echo username=x-oauth; echo password=$GITEE_TOKEN; }; f")
            r = sh(["git", "-c", f"credential.helper={helper}", "clone", "--quiet",
                    url, CLONE_DIR],
                   cwd=WORKDIR, timeout=300)
        if r.returncode != 0:
            print(f"clone failed: {r.stderr[-300:]}", flush=True)
            return False
    return True


def fetch_pr_head(number, head_ref, head_repo, token):
    """Fetch the PR head into local branch ci/pr-<n>. Returns sha or None."""
    env = dict(os.environ)
    if token:
        env["GITEE_TOKEN"] = token
    helper = ("!f() { echo username=x-oauth; echo password=$GITEE_TOKEN; }; f"
              if token else "")
    def git(*args):
        cmd = ["git"]
        if helper:
            cmd += ["-c", f"credential.helper={helper}"]
        cmd += list(args)
        return subprocess.run(cmd, cwd=CLONE_DIR, capture_output=True,
                              text=True, env=env, timeout=180)

    br = f"ci/pr-{number}"
    # same-repo branch is the common case; pull/N/head as fallback, fork last
    for remote_ref in (f"+refs/heads/{head_ref}:refs/heads/{br}",
                       f"+pull/{number}/head:refs/heads/{br}"):
        r = git("fetch", "--quiet", "origin", remote_ref)
        if r.returncode == 0:
            return git("rev-parse", br).stdout.strip() or None
    if head_repo and head_repo.get("path") and head_repo["path"] != REPO_SLUG:
        fork_url = f"https://gitee.com/{head_repo['path']}.git"
        r = git("fetch", "--quiet", fork_url, f"+refs/heads/{head_ref}:refs/heads/{br}")
        if r.returncode == 0:
            return git("rev-parse", br).stdout.strip() or None
    return None


def run_l1(sha):
    os.makedirs(WORKDIR, exist_ok=True)
    if os.path.isdir(WT_DIR):
        sh(["git", "worktree", "remove", "--force", WT_DIR], cwd=CLONE_DIR)
    r = sh(["git", "worktree", "add", "--detach", WT_DIR, sha], cwd=CLONE_DIR)
    if r.returncode != 0:
        return 3, "worktree add failed:\n" + r.stderr[-400:]
    g = sh(["bash", "ci/l1_gpu_gate.sh"], cwd=WT_DIR, timeout=3000)
    tail = "\n".join((g.stdout + g.stderr).splitlines()[-20:])
    return g.returncode, tail


# ---------------------------------------------------------------------------
def sweep(token, comment=True):
    prs = api_get(f"/repos/{REPO_SLUG}/pulls?state=open&per_page=20", token)
    if not isinstance(prs, list):
        print(f"pull list failed: {prs}", flush=True)
        return
    state = load_state()
    for pr in prs:
        num = pr.get("number")
        head = pr.get("head", {}) or {}
        sha = head.get("sha") or ""
        title = (pr.get("title") or "")[:60]
        if not num or not sha:
            continue
        key = str(num)
        rec = state.get(key) or {}
        if rec.get("sha") == sha and rec.get("status") in ("pass", "fail"):
            continue  # already tested this head
        print(f"PR #{num} '{title}' @ {sha[:8]}: testing", flush=True)
        if not ensure_clone(token):
            continue
        got = fetch_pr_head(num, head.get("ref", ""), head.get("repo"), token)
        if got != sha and got is None:
            state[key] = {"sha": sha, "status": "error", "note": "fetch failed"}
            save_state(state)
            continue
        rc, tail = run_l1(got or sha)
        if rc == 2:
            print("  no free GPU -- queued, retry next sweep", flush=True)
            if comment and rec.get("queued_notified_sha") != sha:
                api_post_comment(num, f"⏳ CI L1 queued @ {sha[:8]}: 目前无空闲 GPU，稍后自动重试", token)
                rec["queued_notified_sha"] = sha
                state[key] = {**rec, "sha": rec.get("sha"), "status": rec.get("status", "new")}
                save_state(state)
            continue
        status = "pass" if rc == 0 else "fail"
        icon = "✅" if rc == 0 else "❌"
        state[key] = {"sha": sha, "status": status, "rc": rc}
        save_state(state)
        print(f"  L1 {status} (rc={rc})", flush=True)
        if comment:
            body = (f"{icon} **CI L1 GPU gate {status.upper()}** @ `{sha[:8]}`\n"
                    f"```\n{tail}\n```")
            api_post_comment(num, body, token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single sweep then exit")
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--no-comment", action="store_true")
    args = ap.parse_args()

    token = load_token()
    if not token:
        print("NOTE: no GITEE_TOKEN (~/.hcu_ci_token) -- anonymous API only, "
              "cannot post PR comments", flush=True)
    while True:
        try:
            sweep(token, comment=not args.no_comment)
        except Exception as e:
            print(f"sweep error: {e}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
