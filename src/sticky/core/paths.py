from __future__ import annotations

from pathlib import Path
from typing import Collection, Optional


def is_nullish(value: object, *, nullish: Collection[object]) -> bool:
    return value in nullish


def resolve_against_original_cwd(
    path_like: str,
    *,
    fallback_to_resolve: bool = True,
) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path

    if not bool(fallback_to_resolve):
        import hydra

        return Path(hydra.utils.get_original_cwd()) / path

    try:
        import hydra

        return Path(hydra.utils.get_original_cwd()) / path
    except Exception:
        return path.resolve()


def resolve_optional_path(
    path_like: object,
    *,
    nullish: Collection[object],
    fallback_to_resolve: bool = True,
) -> Optional[Path]:
    if is_nullish(path_like, nullish=nullish):
        return None
    return resolve_against_original_cwd(
        str(path_like),
        fallback_to_resolve=bool(fallback_to_resolve),
    )


def resolve_optional_path_str(
    path_like: object,
    *,
    nullish: Collection[object],
    fallback_to_resolve: bool = True,
) -> Optional[str]:
    resolved = resolve_optional_path(
        path_like,
        nullish=nullish,
        fallback_to_resolve=bool(fallback_to_resolve),
    )
    if resolved is None:
        return None
    return str(resolved)

