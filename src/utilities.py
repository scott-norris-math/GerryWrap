import os


def ensure_directory_exists(output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)


def union(x: list, y: list) -> list:
    return list(set(x) | set(y))