from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.check_public_tree import audit_tree


def test_public_tree_audit_finds_secret_paths_private_ips_and_broken_links():
    with TemporaryDirectory(dir=Path(__file__).parents[1]) as temp_dir:
        root = Path(temp_dir)
        (root / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")
        fake_credential = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        local_path = "C:" + "\\Users" + "\\person\\secret.txt"
        private_ip = ".".join(("10", "2", "3", "4"))
        (root / "config.txt").write_text(
            f"{fake_credential}\n{local_path}\n{private_ip}\n",
            encoding="utf-8",
        )

        issues = audit_tree(root)

    assert any("credential-like" in issue for issue in issues)
    assert any("workstation-local" in issue for issue in issues)
    assert any("private IP" in issue for issue in issues)
    assert any("missing local link" in issue for issue in issues)


def test_public_tree_audit_skips_gitignored_private_documents():
    repo_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix=".public-tree-audit-", dir=repo_root) as temp_dir:
        fixture = Path(temp_dir)
        (fixture / ".gitignore").write_text("private-notes.md\n", encoding="utf-8")
        private_ip = ".".join(("10", "2", "3", "4"))
        (fixture / "private-notes.md").write_text(f"Private IP: {private_ip}\n", encoding="utf-8")

        issues = audit_tree(repo_root)

    assert issues == []


def test_scanner_flags_anthropic_style_key():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        fake_key = "sk-ant-api03-" + "A" * 40
        (root / "provider.txt").write_text(fake_key, encoding="utf-8")

        issues = audit_tree(root)

    assert any("credential-like" in issue for issue in issues)


def test_scanner_still_flags_deepseek_style_key():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        fake_key = "sk-" + "B" * 32
        (root / "provider.txt").write_text(fake_key, encoding="utf-8")

        issues = audit_tree(root)

    assert any("credential-like" in issue for issue in issues)
