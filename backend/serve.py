"""Start the API with a bind address that is safe by default.

    python -m backend.serve                 # 127.0.0.1:8000 — this machine only
    python -m backend.serve --port 8001
    VENUS_BIND_ALL=true python -m backend.serve      # 0.0.0.0, deliberate
    python -m backend.serve --bind-all               # same, from the command line

The demo runs on a laptop on conference or hospital wi-fi, and the API has no
authentication (docs/PRIVACY.md §6): anything that can reach the port can read
every stored screening. So loopback is the default and exposing the port to the
network is an explicit act that prints a warning naming what it exposes.

The container is the one place 0.0.0.0 is normal — a bind to 127.0.0.1 inside a
container is unreachable even from the host — so backend/Dockerfile sets
VENUS_BIND_ALL=true, and that line is the audit trail.
"""

from __future__ import annotations

import argparse
import os
import sys

LOOPBACK = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - deliberate, gated below


def chosen_host(bind_all_flag: bool = False, env: dict | None = None) -> tuple[str, str | None]:
    """(host, warning). Loopback unless explicitly overridden."""
    env = os.environ if env is None else env
    env_opt_in = env.get("VENUS_BIND_ALL", "").strip().lower() in {"1", "true", "yes", "on"}
    if bind_all_flag or env_opt_in:
        why = "--bind-all" if bind_all_flag else "VENUS_BIND_ALL"
        return ALL_INTERFACES, (
            f"listening on every interface ({why}). The API has no authentication: anyone who can reach "
            f"this port can read every stored screening and every report. Use only on a trusted network."
        )
    return LOOPBACK, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=int(os.getenv("VENUS_PORT", "8000")))
    parser.add_argument("--bind-all", action="store_true",
                        help="listen on 0.0.0.0 instead of 127.0.0.1 (no authentication: trusted networks only)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    host, warning = chosen_host(args.bind_all)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Venus AI API on http://{host}:{args.port}  (docs at /docs)")

    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
