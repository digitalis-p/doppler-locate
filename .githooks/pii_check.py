#!/usr/bin/env python3
"""Keep personal data out of git history. Run by the hooks in this directory.

Checks
  identity  author and committer e-mail must be a GitHub noreply address
  text      added lines and commit messages: e-mail addresses (other than noreply ones),
            home-directory paths, credentials, entries of the local deny list, and (unless the
            line carries the marker ``pii-ok``) precise coordinate pairs and IPv4 addresses
  images    text read by OCR (tesseract, when installed) and the file's metadata: same rules
            except coordinates; any GPS metadata is refused
  files     nothing under data/, out/ or cache/, no audio, video or data files, nothing > 5 MB

The deny list (.git/info/pii-denylist: one literal per line, # comments, matched without
regard to case) lives inside .git, so it is never committed. Put private strings there:
an e-mail address, a real name, a street, a host name.

  pii_check.py staged        pre-commit: the staged changes and the commit identity
  pii_check.py msg FILE      commit-msg
  pii_check.py push REMOTE   pre-push: every commit being pushed (ref lines on stdin)
  pii_check.py tree [REV]    audit every file of a revision (default HEAD)

Enable once per clone: git config core.hooksPath .githooks
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
EMAIL_OK = re.compile(r"(@users\.noreply\.github\.com|^noreply@anthropic\.com|^noreply@github\.com"
                      r"|^git@github\.com|@example\.(?:com|org|net))$", re.I)
HOME = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\+Users\\+)(?!user\b|username\b|you\b|<)[A-Za-z0-9._-]+")
SECRETS = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:ant-)?[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AIza[0-9A-Za-z_-]{35}|(?i:(?:password|passwd|secret|api_?key|token)\s*[:=]\s*['\"][^'\"\s]{6,})")
COORD = re.compile(r"(?<![\w.])(-?\d{1,2}\.\d{5,})\s*[,; ]\s*(-?\d{1,3}\.\d{5,})(?![\w.])")
IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")
MARKER = "pii-ok"

DATA_DIRS = ("data/", "out/", "cache/", ".venv/")
DATA_EXT = (".webm", ".mp4", ".mkv", ".mov", ".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg", ".aac",
            ".parquet", ".npy", ".npz", ".tif", ".tiff", ".tar", ".grib2", ".zip", ".7z")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
MAX_BYTES = 5 * 1024 * 1024
ZERO = "0" * 40


def blob_of(*args: str, check: bool = True) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=check).stdout


def git(*args: str, check: bool = True) -> str:
    return blob_of(*args, check=check).decode("utf-8", "replace")


def denylist() -> list[str]:
    path = git("rev-parse", "--git-path", "info/pii-denylist").strip()
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.lower())
    return out


DENY = denylist()


def scan_text(text: str, where: str, coords: bool = True) -> list[str]:
    """Problems in ``text``; ``where`` labels them (file:line, commit message, ...)."""
    bad = []
    low = text.lower()
    for d in DENY:
        if d in low:
            bad.append(f"{where}: deny-list entry")
    for m in EMAIL.finditer(text):
        if not EMAIL_OK.search(m.group(0)):
            bad.append(f"{where}: e-mail address {m.group(0)!r}")
    for m in HOME.finditer(text):
        bad.append(f"{where}: home-directory path {m.group(0)!r}")
    for m in SECRETS.finditer(text):
        bad.append(f"{where}: possible credential {m.group(0)[:12]!r}...")
    if coords and MARKER not in text:
        for m in COORD.finditer(text):
            if abs(float(m.group(1))) <= 90 and abs(float(m.group(2))) <= 180:
                bad.append(f"{where}: precise coordinates {m.group(0)!r} (add '{MARKER}' if not a location)")
        for m in IPV4.finditer(text):
            o = [int(x) for x in m.groups()]
            if max(o) <= 255 and o[0] not in (0, 127) and o != [255] * 4:
                bad.append(f"{where}: IP address {m.group(0)!r} (add '{MARKER}' if harmless)")
    return bad


def scan_lines(text: str, path: str, first: int = 1) -> list[str]:
    bad = []
    for i, line in enumerate(text.splitlines(), first):
        bad += scan_text(line, f"{path}:{i}")
    return bad


def scan_image(blob: bytes, path: str) -> list[str]:
    bad = scan_text(blob.decode("latin-1"), f"{path} (metadata)", coords=False)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "img" + os.path.splitext(path)[1].lower())
        with open(f, "wb") as fh:
            fh.write(blob)
        if shutil.which("identify"):
            meta = subprocess.run(["identify", "-verbose", f], capture_output=True, text=True).stdout
            if re.search(r"exif:GPS", meta):
                bad.append(f"{path}: GPS metadata")
        if shutil.which("tesseract"):
            ocr = subprocess.run(["tesseract", f, "stdout"], capture_output=True, text=True).stdout
            bad += scan_text(ocr, f"{path} (text in image)", coords=False)
        else:
            print(f"pii-check: tesseract not installed, text inside {path} was not checked", file=sys.stderr)
    return bad


def scan_file(path: str, blob: bytes) -> list[str]:
    """Whole-file rules plus the content rules for images and binaries."""
    bad = []
    p = path.lower()
    if p.startswith(DATA_DIRS) or p.endswith(DATA_EXT):
        bad.append(f"{path}: data or media file (keep it out of git)")
    if len(blob) > MAX_BYTES:
        bad.append(f"{path}: {len(blob) / 1e6:.1f} MB (over {MAX_BYTES // 2**20} MB)")
    if p.endswith(IMAGE_EXT):
        bad += scan_image(blob, path)
    elif b"\0" in blob[:8000]:
        bad += scan_text(blob.decode("latin-1"), f"{path} (binary)", coords=False)
    return bad


def check_identity(name: str, email: str, who: str) -> list[str]:
    if email.lower().endswith("@users.noreply.github.com"):
        return scan_text(name, f"{who} name", coords=False)
    return [f"{who} e-mail {email!r} is not a GitHub noreply address "
            f"(git config user.email <id>+<login>@users.noreply.github.com)"]


def ident(var: str) -> tuple[str, str]:
    m = re.match(r"(.*) <(.*)> \d+ [+-]\d{4}$", git("var", var).strip())
    return (m.group(1), m.group(2)) if m else ("?", "?")


def added_lines(diff: str) -> list[str]:
    """Problems in the '+' lines of a unified diff (-U0), labelled with their new line number."""
    bad, path, line = [], "?", 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else raw[4:]
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line = int(m.group(1)) if m else 0
        elif raw.startswith("+"):
            bad += scan_text(raw[1:], f"{path}:{line}")
            line += 1
    return bad


def staged() -> list[str]:
    bad = check_identity(*ident("GIT_AUTHOR_IDENT"), "author")
    bad += check_identity(*ident("GIT_COMMITTER_IDENT"), "committer")
    bad += added_lines(git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=ACMR"))
    for path in filter(None, git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").split("\0")):
        bad += scan_file(path, blob_of("show", f":{path}"))
    return bad


def message(text: str, where: str = "commit message") -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return scan_lines(body, where)


def commit(sha: str) -> list[str]:
    short = sha[:7]
    an, ae, cn, ce = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", sha).strip().split("\0")
    bad = check_identity(an, ae, f"{short} author") + check_identity(cn, ce, f"{short} committer")
    bad += message(git("show", "-s", "--format=%B", sha), f"{short} message")
    diff = git("show", "--format=", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=ACMR", "--first-parent",
               "-m", sha)
    bad += [f"{short} {b}" for b in added_lines(diff)]
    names = git("show", "--format=", "--name-only", "-z", "--diff-filter=ACMR", "--first-parent", "-m", sha)
    for path in filter(None, names.split("\0")):
        bad += [f"{short} {b}" for b in scan_file(path, blob_of("show", f"{sha}:{path}"))]
    return bad


def push(remote: str) -> list[str]:
    bad = []
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:
            continue
        local, remote_sha = parts[1], parts[3]
        rng = None
        if remote_sha != ZERO and git("cat-file", "-t", remote_sha, check=False).strip() == "commit":
            rng = [f"{remote_sha}..{local}"]
        else:
            rng = [local, "--not", f"--remotes={remote}"]
        for sha in git("rev-list", *rng).split():
            bad += commit(sha)
    return bad


def tree(rev: str) -> list[str]:
    bad = []
    for path in filter(None, git("ls-tree", "-r", "-z", "--name-only", rev).split("\0")):
        blob = blob_of("show", f"{rev}:{path}")
        bad += scan_file(path, blob)
        if b"\0" not in blob[:8000] and not path.lower().endswith(IMAGE_EXT):
            bad += scan_lines(blob.decode("utf-8", "replace"), path)
    return bad


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "staged"
    if mode == "staged":
        bad = staged()
    elif mode == "msg":
        bad = message(open(sys.argv[2], encoding="utf-8").read())
    elif mode == "push":
        bad = push(sys.argv[2] if len(sys.argv) > 2 else "origin")
    elif mode == "tree":
        bad = tree(sys.argv[2] if len(sys.argv) > 2 else "HEAD")
    else:
        print(__doc__)
        return 2
    if bad:
        print(f"pii-check ({mode}): blocked, {len(bad)} problem(s):", file=sys.stderr)
        for b in dict.fromkeys(bad):
            print(f"  {b}", file=sys.stderr)
        print("Remove them (rewrite the commit if it is already made). Do not bypass with --no-verify.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
