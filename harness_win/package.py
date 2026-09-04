import argparse
import zipfile
from collections.abc import Iterator
from pathlib import Path

DEFAULT_INCLUDES = ("agent.py", "requirements.txt", "weights")
SKIP = {"__pycache__", ".DS_Store"}


def members(root: Path, includes: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    for name in includes:
        source = root / name
        if source.is_file():
            yield source, name
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not SKIP & set(path.parts):
                    yield path, str(path.relative_to(root))


def build(root: Path, destination: Path, includes: tuple[str, ...]) -> list[str]:
    written: list[str] = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in members(root, includes):
            archive.write(source, name)
            written.append(name)
    if "agent.py" not in written:
        raise SystemExit(f"{root / 'agent.py'} does not exist; the platform imports it by name")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission zip.")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--include", action="append", default=[])
    arguments = parser.parse_args()

    includes = DEFAULT_INCLUDES + tuple(arguments.include)
    written = build(Path.cwd(), arguments.out, includes)
    size = arguments.out.stat().st_size
    print(f"{arguments.out} ({size:,} bytes)")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
