import sys

from .experiments import drift, phase_diagram, static_baseline

COMMANDS = {
    "static": "R-H1: simulator against the stationary analytic model",
    "drift": "policy costs across drift and recurrence, plus the written predictions",
    "phase": "ASCII regime maps under several cost models",
    "all": "run everything",
}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0] if args else ""
    if cmd not in COMMANDS:
        print("usage: python -m muc.economics {" + ",".join(COMMANDS) + "}")
        for name, blurb in COMMANDS.items():
            print(f"  {name:<7}{blurb}")
        return 0 if not args else 2
    if cmd in ("static", "all"):
        static_baseline.main()
    grid = None
    if cmd in ("drift", "all"):
        grid = drift.main()
    if cmd in ("phase", "all"):
        phase_diagram.main(grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
