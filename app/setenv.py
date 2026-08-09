import argparse
import os
import sys
import shutil
from pathlib import Path

# --- CONFIGURATION ---
INTEL_INDEX = "https://software.repos.intel.com/python/pypi"

INTEL_FIX_CONTENT = """\
# _distributor_init_local.py
# Copy me to venv/Lib/site-packages/numpy if you use Intel's numpy distribution.
# When this is a frozen system (PyInstaller), I assume that you copied the dll's 
# to the executable directory. 
# Otherwise they are assumed to be in the default Intel location.
import os, sys
if getattr(sys, "frozen", False):
    os.add_dll_directory(sys.prefix)
else:
    os.add_dll_directory(os.path.join(sys.prefix, 'Library', 'bin'))
"""


def resolve_venv_path(project_root: Path) -> Path:
    """ Determine the venv path
    """
    actual = os.environ.get("UV_PROJECT_ENVIRONMENT")
    return Path(actual) if actual else project_root / ".venv"


def setup(project_root) -> None:
    """ Main setup function.
        If numpy is installed from Intel index we need to set _distributer_local to let numpy find its dll's
        Normally this would be done by the Intel distribution installer, but we use uv, and only a few
        packages (numpy, scipy). So we have to write _distributer_local  ourselves.
        Using Intel's packages is strongly recommended, they are faster and more stable.
    """
    # toml is the single source of truth must exist
    toml = project_root / "pyproject.toml"
    if not toml.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found in {project_root}. "
        )

    # (Over)Write Intel distributor file if toml declares the Intel index
    toml_text = toml.read_text(encoding="utf-8")
    if 'name = "intel"' in toml_text:
        target_venv = resolve_venv_path(project_root)
        numpy_path = target_venv / "Lib" / "site-packages" / "numpy"
        if numpy_path.exists():
            dil_path = numpy_path / "_distributor_init_local.py"            
            dil_path.write_text(INTEL_FIX_CONTENT, encoding="utf-8")
            print("[FIX] Written Intel distributor fix to numpy.")


def main() -> None:
    project_root = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(
        prog="setenv.py",
        description="Python venv tool patches intel numpy if installed from pyproject.toml.",
    )
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "-p", "--print-venv-path",
        action="store_true",
        default=False,
        help="Print the resolved venv path and exit.",
    )

    parser.add_argument(
        "-v", "--print-version",
        action="store_true",
        default=False,
        help="Print the python version and exit.",
    )
       
    args = parser.parse_args()

    try:
        if args.print_venv_path:
            # make sure to print without newline for a batch file to capture it correctly
            print(resolve_venv_path(project_root), end ="")
            return

        if args.print_version:
            # make sure to print without newline for a batch file to capture it correctly
            import platform
            print(f"Python {platform.python_version()}", end ="")
            return
        #
        setup(project_root)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

