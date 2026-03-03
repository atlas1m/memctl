"""License key system for memctl Pro features."""

import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".memctl" / "config.toml"
LICENSE_KEY_ENV = "MEMCTL_LICENSE_KEY"


def get_license_key() -> str | None:
    """Return the license key from env or config file."""
    # 1. Check environment variable first
    key = os.environ.get(LICENSE_KEY_ENV)
    if key:
        return key.strip()

    # 2. Check config.toml
    if not CONFIG_PATH.exists():
        return None

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        return data.get("license", {}).get("key")
    except Exception:
        return None


def is_pro() -> bool:
    """Return True if a valid license key is configured."""
    key = get_license_key()
    return bool(key and len(key) > 8)


def set_license_key(key: str) -> None:
    """Persist the license key in config.toml."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config
    existing = {}
    if CONFIG_PATH.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(CONFIG_PATH, "rb") as f:
                existing = tomllib.load(f)
        except Exception:
            pass

    # Update license section
    existing.setdefault("license", {})["key"] = key

    # Write back (simple TOML serialization)
    lines = []
    for section, values in existing.items():
        lines.append(f"\n[{section}]")
        for k, v in values.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")

    CONFIG_PATH.write_text("\n".join(lines).strip() + "\n")


def require_pro(feature_name: str) -> bool:
    """Check if Pro is active. Returns True if OK, False if feature is locked."""
    if is_pro():
        return True
    return False


PRO_UPGRADE_MSG = (
    "\n[yellow]⚡ This feature requires memctl Pro.[/yellow]\n"
    "  • Memory decay curves\n"
    "  • LLM consolidation\n"
    "  • memctl health diagnostics\n\n"
    "  [bold]Upgrade:[/bold] https://polar.sh/atlas1m\n"
    "  Then run: [bold]memctl license set <your-key>[/bold]\n"
)
