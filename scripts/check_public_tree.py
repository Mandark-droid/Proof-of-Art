"""Scan a publishable tree and, optionally, its candidate public Git history."""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", ".runtime-packages", "__pycache__", ".pytest_cache", "results"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".txt", ".gitignore"}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"sk-ant-api\d{2}-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?:sk-[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})"),
)
LOCAL_PATH = re.compile(r"(?:C:\\Users\\|D:\\Projects\\)", re.IGNORECASE)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
BANNED_PATH = re.compile(r"(?:^|/)(?:docs/(?:social|video|demo)[^/]*|[^/]*draft[^/]*|local)(?:/|$)", re.IGNORECASE)
BANNED_PHRASES = (
    "Sess" + "ion A", "Sess" + "ion B", "Cod" + "ex", "plan" + "ner",
    "#Build" + "WithBeam", "Link" + "edIn", "Dhan" + "teras",
    "CLAUDE_" + "PLAN", "IMPLEMENTATION_" + "PLAN",
    "refused" + " connections", "this" + " environment", "search" + " snippet",
)
NO_REPLY_SUFFIXES = ("@users.noreply.github.com", "@users.noreply.huggingface.co", "@noreply.github.com")


def _publishable_files(root: Path):
    root = root.resolve()
    try:
        git_root = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_root = ""

    if git_root and Path(git_root).resolve() == root:
        try:
            listing = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                check=True, capture_output=True,
            ).stdout
            candidates = (root / item.decode("utf-8", errors="surrogateescape")
                          for item in listing.split(b"\0") if item)
        except (OSError, subprocess.CalledProcessError):
            candidates = root.rglob("*")
    else:
        candidates = root.rglob("*")

    for path in candidates:
        if (not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(root).parts)
                or path.name == ".env"):
            continue
        yield path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _audit_history(root: Path) -> list[str]:
    issues = []
    try:
        refs = _git(root, "for-each-ref", "--format=%(refname)").splitlines()
        non_main = [ref for ref in refs if ref != "refs/heads/main"]
        if non_main or "refs/heads/main" not in refs:
            issues.append("Git refs must contain only refs/heads/main")
        commits = _git(root, "log", "--format=%H%x09%an%x09%ae%x09%cn%x09%ce", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return ["candidate history is not a readable Git repository"]
    if not commits:
        return issues + ["candidate history has no commits"]
    for record in commits:
        fields = record.split("\t")
        if len(fields) != 5:
            issues.append("could not parse commit identity metadata")
            continue
        commit, author, author_email, committer, committer_email = fields
        for role, name, email in (("author", author, author_email), ("committer", committer, committer_email)):
            if not email.lower().endswith(NO_REPLY_SUFFIXES):
                issues.append(f"commit {commit[:12]} {role} email is not a no-reply address")
            if EMAIL.search(name):
                issues.append(f"commit {commit[:12]} {role} name contains an email address")
    return issues


def audit_tree(root: Path = ROOT, *, check_history: bool = False) -> list[str]:
    root = root.resolve()
    issues = []
    files = list(_publishable_files(root))
    for path in files:
        relative = path.relative_to(root).as_posix()
        if BANNED_PATH.search(relative):
            issues.append(f"{relative}: banned public path")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(f"{relative}: credential-like string")
                break
        if LOCAL_PATH.search(text):
            issues.append(f"{relative}: workstation-local absolute path")
        if EMAIL.search(text):
            issues.append(f"{relative}: email address")
        for match in IPV4.findall(text):
            try:
                address = ipaddress.ip_address(match)
            except ValueError:
                continue
            if address.is_private:
                issues.append(f"{relative}: private IP address")
                break
        for phrase in BANNED_PHRASES:
            if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.IGNORECASE):
                issues.append(f"{relative}: banned phrase {phrase!r}")
                break
        if path.suffix.lower() == ".md":
            for target in MARKDOWN_LINK.findall(text):
                target = target.strip().strip("<>")
                if not target or target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                local_target = target.split("#", 1)[0].split("?", 1)[0]
                if local_target and not (path.parent / local_target).exists():
                    issues.append(f"{relative}: missing local link target {local_target}")
    if check_history:
        issues.extend(_audit_history(root))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--history", action="store_true", help="also enforce one main ref and no-reply commit identities")
    args = parser.parse_args()
    issues = audit_tree(args.root, check_history=args.history)
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 1
    suffix = " and candidate history" if args.history else ""
    print(f"PASS public tree{suffix}: no banned paths or phrases, recognized secrets, local paths, private IPs, emails, or broken local Markdown links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
