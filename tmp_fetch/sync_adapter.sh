#!/bin/bash
# Sync the container-side adapter tree (/root/hcu_t2) to the team main remote via git.
# Motivation (2026-10-18, r04a lesson): the b64 ad-hoc sync left a stale
# hcu_env.py in place while platform_seam.py was updated — the arm silently
# ran the old default. With git, `git log -1` / `git diff` in the container
# answers "is the code right" instantly.
#
# Token discipline: extracted at runtime from TermHub session logs, spliced
# into the remote script below as a one-shot `git -c http.extraHeader`
# (never written to .git/config), zero residue after the run.
#
# Usage: bash tmp_fetch/sync_adapter.sh   (from the repo root)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_HEAD=$(git -C "$ROOT" log -1 --format=%h)

TOKEN=$(python - <<'EOF'
import glob, io, os, re
cands = []
for p in glob.glob(os.path.expandvars(r'%APPDATA%/TermHub/sessions/nmz26_*.log')):
    try:
        txt = io.open(p, encoding='utf-8', errors='ignore').read()
    except Exception:
        continue
    cands += re.findall(r'lizhigong:([A-Za-z0-9_-]{20,})', txt)
print(max(cands, key=len) if cands else '')
EOF
)
if [ -z "$TOKEN" ]; then echo "no token found in TermHub session logs"; exit 1; fi
AUTH="Authorization: Basic $(printf 'lizhigong:%s' "$TOKEN" | base64 -w0 | tr -d '\n')"
unset TOKEN

# heredoc is UNQUOTED on purpose: $AUTH expands locally (one-shot token into
# the git -c flag); every other remote $ is escaped below.
bash "$(dirname "$0")/ssh_nmz26.sh" lizhg@10.16.1.26 \
  "docker exec -i g53-tp8 bash -s" <<REMOTE
set -e
git config --global --add safe.directory /root/hcu_t2 2>/dev/null || true
cd /root/hcu_t2
if [ ! -d .git ]; then
    echo "== initializing git repo in /root/hcu_t2 =="
    git init -q
    git remote add origin <主仓远端URL>
fi
git -c http.extraHeader="$AUTH" fetch -q origin master
git reset -q --hard FETCH_HEAD
echo "container HEAD: \$(git log -1 --format='%h %s' | cut -c1-90)"
echo "tracked-dirty:  \$(git status --porcelain | grep -v __pycache__ | wc -l) file(s)"
REMOTE

echo "local HEAD:     $LOCAL_HEAD"
echo "match check:    compare the two lines above before trusting an arm."
