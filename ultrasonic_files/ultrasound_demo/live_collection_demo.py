from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_commands(demo_dir, com_port, gain):
    return {
        "spga": [str(demo_dir / "dacq_spga.exe"), str(com_port), str(gain), "normal"],
        "pipe": [str(demo_dir / "dacq2pipe.exe"), str(com_port)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Thin helper for the existing MATLAB ultrasound live-collection demo."
    )
    parser.add_argument("--com", required=True, help="Serial COM port number, for example 5.")
    parser.add_argument("--gain", default="1", help="Receiver gain passed to dacq_spga.exe.")
    parser.add_argument(
        "--matlab-script",
        default="script_us.m",
        help="MATLAB script to run after the acquisition tools are started.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch dacq_spga.exe and dacq2pipe.exe. Without this flag, the script stays in dry-run mode.",
    )
    args = parser.parse_args()

    demo_dir = Path(__file__).resolve().parent
    matlab_script = demo_dir / args.matlab_script
    spga_exe = demo_dir / "dacq_spga.exe"
    pipe_exe = demo_dir / "dacq2pipe.exe"

    missing = [str(path) for path in (spga_exe, pipe_exe, matlab_script) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required live-demo files: {missing}")

    matlab_exe = shutil.which("matlab")
    commands = build_commands(demo_dir, args.com, args.gain)

    print("Live collection demo workflow")
    print(f"Demo directory: {demo_dir}")
    print(f"MATLAB script: {matlab_script.name}")
    print(f"COM port: {args.com}")
    print(f"Gain: {args.gain}")
    print("")
    print("Step 1. Confirm the sensor is connected and the COM port is correct.")
    print(f"Step 2. Start acquisition setup: {' '.join(commands['spga'])}")
    print(f"Step 3. Start pipe reader: {' '.join(commands['pipe'])}")
    print("Step 4. In MATLAB, run:")
    if matlab_exe:
        print(f'  matlab -sd "{demo_dir}" -batch "{matlab_script.stem}"')
    else:
        print(f"  Open MATLAB in {demo_dir} and run {matlab_script.name}")

    if not args.run:
        print("")
        print("Dry run only. Re-run with --run to launch the two Windows executables.")
        return

    print("")
    print("Launching acquisition executables...")
    subprocess.Popen(commands["spga"], cwd=demo_dir)
    subprocess.Popen(commands["pipe"], cwd=demo_dir)
    print("Executables started. Launch MATLAB manually using the command above.")


if __name__ == "__main__":
    main()
