import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    loaded_any = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        if not key:
            continue

        os.environ.setdefault(key, value)
        loaded_any = True

    return loaded_any


def load_local_env(env_path: str | None = None) -> str:
    """
    프로젝트 루트(.env) 또는 지정 경로의 env 파일을 읽어 환경변수를 채운다.
    이미 프로세스에 설정된 환경변수는 덮어쓰지 않는다.

    Returns:
        로드된 env 파일의 절대 경로. 로드하지 못하면 빈 문자열.
    """
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    else:
        candidates.append(Path.cwd() / ".env")
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / ".env")

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _load_env_file(resolved):
            return str(resolved)
    return ""
