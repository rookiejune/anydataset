from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._files import cleanup_store_files
from .migration import migrate_store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anydataset-store")
    commands = parser.add_subparsers(dest="command", required=True)

    migrate = commands.add_parser("migrate", help="migrate a schema-v1 store to v2")
    migrate.add_argument("source")
    migrate.add_argument("output")

    cleanup = commands.add_parser(
        "cleanup-files",
        help="remove extracted AudioView.FILE payloads for one store",
    )
    cleanup.add_argument("store")

    args = parser.parse_args(argv)
    if args.command == "migrate":
        print(migrate_store(args.source, args.output))
        return 0
    removed = cleanup_store_files(args.store)
    print("removed" if removed else "absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
