"""``gist-access``: issue, list, revoke and transfer access codes.

    gist-access create --label "Marvin" [--expires-days 30]
    gist-access create --label "Elvis" --admin
    gist-access list
    gist-access revoke <code id>
    gist-access transfer <from code id> <to code id>

A code is printed once, at creation, and cannot be shown again: only its hash
is stored. A lost code is replaced by creating a new one and transferring the
old one's videos to it. Code ids may be shortened to any unique prefix, as
``list`` prints them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from gist.access import store
from gist.db import connection as db


def _when(value: datetime | None) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC") if value else "-"


def _create(args: argparse.Namespace) -> int:
    expires_at = (
        datetime.now(UTC) + timedelta(days=args.expires_days) if args.expires_days else None
    )
    code, code_id = store.create_code(args.label, is_admin=args.admin, expires_at=expires_at)
    print(f"Code for {args.label}{' (admin)' if args.admin else ''}:\n\n    {code}\n")
    print("Hand this over directly. It is not stored and cannot be shown again.")
    print(f"id {code_id[:8]}, expires {_when(expires_at) if expires_at else 'never'}")
    if args.admin:
        claimed = store.claim_unowned_videos(code_id)
        if claimed:
            print(f"Claimed {claimed} video(s) added before access codes existed.")
    return 0


def _list(_: argparse.Namespace) -> int:
    rows = store.list_codes()
    if not rows:
        print("No codes yet. Create one with: gist-access create --label NAME --admin")
        return 0
    print(f"{'id':<9} {'label':<24} {'role':<6} {'videos':>6} {'sessions':>8}  status")
    for row in rows:
        if row.revoked_at:
            status = f"revoked {_when(row.revoked_at)}"
        elif row.expires_at and row.expires_at <= datetime.now(UTC):
            status = f"expired {_when(row.expires_at)}"
        elif row.expires_at:
            status = f"active until {_when(row.expires_at)}"
        else:
            status = "active"
        role = "admin" if row.is_admin else "user"
        print(
            f"{row.id[:8]:<9} {row.label[:24]:<24} {role:<6} "
            f"{row.video_count:>6} {row.active_sessions:>8}  {status}"
        )
    return 0


def _revoke(args: argparse.Namespace) -> int:
    code_id = store.resolve_code_id(args.code_id)
    store.revoke_code(code_id)
    print(f"Revoked {code_id[:8]} and ended its sessions. Its videos are kept.")
    return 0


def _transfer(args: argparse.Namespace) -> int:
    source = store.resolve_code_id(args.from_code_id)
    target = store.resolve_code_id(args.to_code_id)
    moved = store.transfer_videos(source, target)
    print(f"Moved {moved} video(s) from {source[:8]} to {target[:8]}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gist-access", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="issue a new code")
    create.add_argument("--label", required=True, help="who the code is for")
    create.add_argument("--admin", action="store_true", help="sees all videos, no daily cap")
    create.add_argument("--expires-days", type=float, help="stop working after this many days")
    create.set_defaults(run=_create)

    commands.add_parser("list", help="show every code").set_defaults(run=_list)

    revoke = commands.add_parser("revoke", help="disable a code and sign it out everywhere")
    revoke.add_argument("code_id")
    revoke.set_defaults(run=_revoke)

    transfer = commands.add_parser("transfer", help="move one code's videos to another")
    transfer.add_argument("from_code_id")
    transfer.add_argument("to_code_id")
    transfer.set_defaults(run=_transfer)

    args = parser.parse_args(argv)
    load_dotenv()
    if not db.is_configured():
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 2
    db.apply_schema()
    try:
        return args.run(args)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        db.close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
