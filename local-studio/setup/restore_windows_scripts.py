"""Rename *.ps1.txt back to *.ps1.

Gmail blocks attachments containing PowerShell scripts, so the emailed copy of
this package ships them as .ps1.txt. Run once after unzipping:

    python setup/restore_windows_scripts.py
"""

from pathlib import Path

root = Path(__file__).resolve().parent.parent
renamed = 0
for path in root.rglob("*.ps1.txt"):
    target = path.with_suffix("")  # drops .txt
    if target.exists():
        print(f"skip (already exists): {target.relative_to(root)}")
        continue
    path.rename(target)
    print(f"restored {target.relative_to(root)}")
    renamed += 1
print(f"{renamed} script(s) restored.")
