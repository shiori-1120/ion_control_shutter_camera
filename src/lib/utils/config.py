from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _yaml_available = True
except Exception:
    _yaml_available = False

class ConfigError(Exception):
    pass

DEFAULT_ENCODING = "utf-8"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    if not _yaml_available:
        raise ConfigError("PyYAML is not installed. Run: pip install pyyaml")
    text = p.read_text(encoding=DEFAULT_ENCODING)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid YAML root type in {p}: {type(data)}")
    return data


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    if not _yaml_available:
        raise ConfigError("PyYAML is not installed. Run: pip install pyyaml")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, indent=2)
    p.write_text(text, encoding=DEFAULT_ENCODING)


def resolve_visa_from_profiles(cfg: Dict[str, Any]) -> Optional[str]:
    profiles = cfg.get('visa_profiles')
    current = cfg.get('current_profile')
    if isinstance(profiles, dict) and isinstance(current, str):
        prof = profiles.get(current)
        if isinstance(prof, dict):
            res = prof.get('resource')
            if isinstance(res, str) and res:
                return res
    return None
