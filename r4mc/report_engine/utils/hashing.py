"""File hashing utilities for manifest generation."""
import hashlib
from pathlib import Path
from typing import Dict, Union


def hash_file(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of file hash
    """
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(dir_path: Union[str, Path], algorithm: str = 'sha256') -> Dict[str, str]:
    """
    Compute hashes for all files in a directory.

    Args:
        dir_path: Path to directory
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Dict mapping relative file paths to hashes
    """
    dir_path = Path(dir_path)
    hashes = {}

    for file_path in sorted(dir_path.rglob('*')):
        if file_path.is_file():
            rel_path = file_path.relative_to(dir_path)
            hashes[str(rel_path)] = hash_file(file_path, algorithm)

    return hashes


def hash_string(s: str, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a string.

    Args:
        s: Input string
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of string hash
    """
    h = hashlib.new(algorithm)
    h.update(s.encode('utf-8'))
    return h.hexdigest()
