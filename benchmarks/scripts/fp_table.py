#!/usr/bin/env python3
"""One row per (Func, Type, Accuracy) from the CSV output of the fpmp benchmark executables
(libcudacxx/benchmarks/bench/fp/fpmp/*.cu), with the SASS instruction count of one evaluation
joined in from each executable.

    fp_table.py --exe bin/libcudacxx.bench.fp.fpmp.arith.base --csv out/arith.csv \\
                --exe bin/libcudacxx.bench.fp.fpmp.explog.base --csv out/explog.csv ... \\
                [--out table.csv] [--md table.md]

or, with the executables in one directory and the CSVs named after them in another:

    fp_table.py --bin-dir build/bin --csv-dir out [--out table.csv] [--md table.md]

Each executable compiles a __noinline__ function `fpb::bare<Op, cuda::fpmp2<L, A>>` per cell.
cuobjdump extracts the cubins and nvdisasm lists every kernel's section with the local functions
it calls as labelled blocks (`$<kernel>$<function>:`); the SASS column is the number of
instructions of the bare function's block, label to RET, i.e. the instructions one evaluation of
the operation takes, without its CALL/RET pair and without the kernel's loads and stores. Where a
cubin holds several architectures the largest count is kept.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ACC = {"1": "low", "2": "mid", "3": "high"}
TYP = {"float": "fp32mp2", "double": "fp64mp2"}
BARE_RE = re.compile(
    r"fpb::bare<fpb::op_(\w+), cuda::(?:__\d+::)?fpmp2<(float|double), \(cuda::(?:__\d+::)?fpmp2_accuracy\)(\d)>"
)
COLS = [
    "Func",
    "Type",
    "Accuracy",
    "u2",
    "Bits",
    "ev/clk/SM",
    "clk/ev",
    "SASS",
    "Regs",
    "Elem/s",
    "GPU Time (s)",
    "Worst at",
    "Family",
    "Device",
]
ACC_ORDER = {"low": 0, "mid": 1, "high": 2}


def sass_counts(exe):
    """{(func, type, accuracy): instruction count} of the bare functions in `exe`.

    cuobjdump extracts the cubins and nvdisasm lists each kernel's section: the kernel's own code
    first, then the local functions it calls as labelled blocks (`$<kernel>$<function>:`), each
    split further by branch labels (`.L_x_N:`). In a bare kernel's section everything after the
    bare function's label is the operation: its own body, its out-of-line paths and any fpmp
    helper the compiler kept as a function. Not counted: NOP padding, RET, the self-loop BRA that
    ends a section, and CUDA's own `__cuda_*` subroutines (the suite's rule as well).
    """
    bindir = os.path.dirname(
        shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    )
    filt = shutil.which("c++filt")

    def demangle(names):
        if not filt or not names:
            return list(names)
        return subprocess.run(
            [filt], input="\n".join(names), capture_output=True, text=True
        ).stdout.splitlines()

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [os.path.join(bindir, "cuobjdump"), "-xelf", "all", os.path.abspath(exe)],
            check=True,
            capture_output=True,
            cwd=tmp,
        )
        for cubin in sorted(os.listdir(tmp)):
            if not cubin.endswith(".cubin"):
                continue
            dis = subprocess.run(
                [os.path.join(bindir, "nvdisasm"), "-c", os.path.join(tmp, cubin)],
                capture_output=True,
                text=True,
            ).stdout
            # pass 1: the sections of the bare kernels and the local labels they contain
            sections = {}  # section symbol -> list of lines
            cur = None
            for line in dis.splitlines():
                m = re.match(r"\t\.section\t\.text\.(\S+?),", line)
                if m:
                    cur = m.group(1) if "bare_kernel" in m.group(1) else None
                    if cur:
                        sections[cur] = []
                    continue
                if cur:
                    sections[cur].append(line)
            if not sections:
                continue
            kernels = list(sections)
            locals_ = sorted(
                {
                    m.group(1)
                    for lines in sections.values()
                    for text in lines
                    for m in [re.match(r"^\$\S+\$(\S+):\s*$", text)]
                    if m
                }
            )
            pretty = dict(zip(kernels + locals_, demangle(kernels + locals_)))
            # pass 2: count
            for sym, lines in sections.items():
                m = BARE_RE.search(pretty[sym].replace("bare_kernel", "bare"))
                if not m:
                    continue
                key = (m.group(1), TYP[m.group(2)], ACC[m.group(3)])
                kind, label, count = "kernel", None, 0
                for line in lines:
                    lm = re.match(r"^\$\S+\$(\S+):\s*$", line)
                    if lm:
                        name = pretty.get(lm.group(1), lm.group(1))
                        kind = (
                            "cuda"
                            if name.startswith("__cuda")
                            else ("bare" if "fpb::bare<" in name else "helper")
                        )
                        label = lm.group(1)
                        continue
                    lm = re.match(r"^(\.L_x_\d+):\s*$", line)
                    if lm:
                        label = lm.group(1)
                        continue
                    if re.match(r"^\S+:\s*$", line):  # the kernel's own symbol
                        kind = "kernel"
                        continue
                    if kind not in ("bare", "helper") or not re.match(
                        r"\s+/\*[0-9a-f]{4,}\*/", line
                    ):
                        continue
                    body = line.split("*/", 1)[1].strip()
                    if body.startswith("NOP") or re.match(r"^RET\b", body):
                        continue
                    if (
                        re.match(r"^BRA\s+`\((\S+)\)", body)
                        and label
                        and re.match(r"^BRA\s+`\((\S+)\)", body).group(1) == label
                    ):
                        continue  # the self-loop after the last RET
                    count += 1
                out[key] = max(out.get(key, 0), count)
    return out


def family_of(exe):
    m = re.search(r"fp\.fpmp\.(\w+)", os.path.basename(exe))
    return m.group(1) if m else os.path.basename(exe)


def merge(exe, csv_path, rows):
    sass = sass_counts(exe)
    fam = family_of(exe)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Skipped", "No") == "Yes":
                continue
            key = (r["Func"], r["Type"], r["Accuracy"])
            row = rows.setdefault(
                key,
                {
                    "Func": key[0],
                    "Type": key[1],
                    "Accuracy": key[2],
                    "Family": fam,
                    "Device": r.get("Device Name", ""),
                },
            )
            if r["Benchmark"] == "throughput":
                row["u2"] = r.get("u2", "")
                row["Bits"] = r.get("Bits", "")
                row["Regs"] = r.get("Regs", "")
                row["ev/clk/SM"] = r.get("ev/clk/SM", "")
                row["Elem/s"] = r.get("Elem/s (elem/sec)", r.get("Elem/s", ""))
                row["GPU Time (s)"] = r.get("GPU Time (sec)", "")
                row["Worst at"] = r.get("Worst at", "")
            elif r["Benchmark"] == "latency":
                row["clk/ev"] = r.get("clk/ev", "")
            row["SASS"] = sass.get(key, "")


def fmt(k, v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v if v is not None else ""
    if k in ("u2", "ev/clk/SM"):
        return f"{x:.2f}"
    if k == "clk/ev":
        return f"{x:.1f}"
    if k in ("Elem/s", "GPU Time (s)"):
        return f"{x:.3e}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exe",
        action="append",
        default=[],
        help="a benchmark executable (repeatable, paired with --csv)",
    )
    ap.add_argument(
        "--csv", action="append", default=[], help="its --csv output (repeatable)"
    )
    ap.add_argument(
        "--bin-dir",
        help="directory with the libcudacxx.bench.fp.fpmp.*.base executables",
    )
    ap.add_argument("--csv-dir", help="directory with <family>.csv for each of them")
    ap.add_argument("--out", help="merged CSV (default: stdout)")
    ap.add_argument("--md", help="also write a markdown table here")
    a = ap.parse_args()

    pairs = list(zip(a.exe, a.csv))
    if len(a.exe) != len(a.csv):
        sys.exit("--exe and --csv must be given in pairs")
    if a.bin_dir and a.csv_dir:
        for exe in sorted(
            glob.glob(os.path.join(a.bin_dir, "libcudacxx.bench.fp.fpmp.*.base"))
        ):
            c = os.path.join(a.csv_dir, family_of(exe) + ".csv")
            if os.path.exists(c):
                pairs.append((exe, c))
    if not pairs:
        sys.exit("nothing to do: give --exe/--csv pairs or --bin-dir and --csv-dir")

    rows = {}
    for exe, c in pairs:
        merge(exe, c, rows)
    table = sorted(
        rows.values(),
        key=lambda r: (
            r["Family"],
            r["Func"],
            r["Type"],
            ACC_ORDER.get(r["Accuracy"], 9),
        ),
    )

    out = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.writer(out)
    w.writerow(COLS)
    for r in table:
        w.writerow([fmt(c, r.get(c, "")) for c in COLS])
    if a.out:
        out.close()
    if a.md:
        md_cols = COLS[:9]
        with open(a.md, "w") as f:
            f.write("| " + " | ".join(md_cols) + " |\n")
            f.write("|" + "---|" * len(md_cols) + "\n")
            for r in table:
                f.write(
                    "| " + " | ".join(fmt(c, r.get(c, "")) for c in md_cols) + " |\n"
                )


if __name__ == "__main__":
    main()
