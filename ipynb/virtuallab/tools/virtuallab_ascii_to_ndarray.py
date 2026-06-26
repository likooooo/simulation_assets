#!/usr/bin/env python3
# vlab_reader.py
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt

# ----------------------------
# Parse a single file:
# returns (meta_dict, comments_string, ndarray)
# meta only contains keys: nx, ny, dx, dy, wavelength, component, data_type
# comments_string: all comment lines concatenated (without leading '#')
# ----------------------------
def parse_file(path):
    comments_lines = []
    data_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.strip().startswith("#"):
                # remove leading '#' and possible leading space
                comments_lines.append(raw.lstrip("#").strip())
            elif raw.strip():
                data_lines.append(raw)

    # build comment string (single concatenated string)
    comments_str = "\n".join(comments_lines)

    # prepare meta with required keys (default None if missing)
    meta = {k: None for k in ("nx", "ny", "dx", "dy", "wavelength", "component", "data_type")}

    header = comments_str

    # Number of sampling points: (100; 100)
    m = re.search(r"Number of sampling points\s*:\s*\(\s*(\d+)\s*[;,]\s*(\d+)\s*\)", header, flags=re.IGNORECASE)
    if m:
        meta["nx"] = int(m.group(1))
        meta["ny"] = int(m.group(2))

    # Sampling distance [mm]: (1E-07; 1E-07)
    m = re.search(r"Sampling distance[^\(]*\(\s*([+\-0-9Ee\.]+)\s*[;,]\s*([+\-0-9Ee\.]+)\s*\)", header, flags=re.IGNORECASE)
    if m:
        try:
            meta["dx"] = float(m.group(1))
            meta["dy"] = float(m.group(2))
        except:
            meta["dx"], meta["dy"] = None, None

    # Wavelength [mm]: 5E-07
    m = re.search(r"Wavelength[^\:]*:\s*([+\-0-9Ee\.]+)", header, flags=re.IGNORECASE)
    if m:
        try:
            meta["wavelength"] = float(m.group(1))
        except:
            meta["wavelength"] = None

    # component: Ex component of Harmonic Field.
    m = re.search(r"(\bEx\b|\bEy\b|\bEz\b|\bHx\b|\bHy\b|\bHz\b)\s+component\s+of\s+Harmonic\s+Field", header, flags=re.IGNORECASE)
    if m:
        meta["component"] = m.group(1)

    # data_type: Strictly parse from "Extracted ..." lines
    # Examples in header: "# Ex component of Harmonic Field.\n# Extracted real part"
    # We'll look for 'Extracted' and take the next token(s).
    m = re.search(r"Extracted\s+([A-Za-z]+)", header, flags=re.IGNORECASE)
    if m:
        token = m.group(1).lower()
        if token in ("real", "realpart", "real_part"):
            meta["data_type"] = "real"
        elif token in ("imag", "imaginary", "imagpart", "imag_part"):
            meta["data_type"] = "imag"
        elif token in ("amplitude", "amp", "magnitude"):
            meta["data_type"] = "amp"
        elif token in ("phase",):
            meta["data_type"] = "phase"

    # fallback: sometimes header contains "Extracted real part" in a different line,
    # so search all comment lines for 'Extracted' and possible tokens
    if meta["data_type"] is None:
        for cl in comments_lines:
            m2 = re.search(r"Extracted\s+([A-Za-z]+)", cl, flags=re.IGNORECASE)
            if m2:
                token = m2.group(1).lower()
                if token in ("real", "realpart", "real_part"):
                    meta["data_type"] = "real"; break
                if token in ("imag", "imaginary", "imagpart", "imag_part"):
                    meta["data_type"] = "imag"; break
                if token in ("amplitude", "amp", "magnitude"):
                    meta["data_type"] = "amp"; break
                if token in ("phase",):
                    meta["data_type"] = "phase"; break

    # -------------------
    # parse numeric data lines into numpy array
    # support separators: whitespace (multi), tab, comma, semicolon
    # -------------------
    rows = []
    for line in data_lines:
        parts = re.split(r"[,\s;]+", line.strip())
        parts = [p for p in parts if p]
        # protect against lines that are not numeric
        try:
            row = [float(p) for p in parts]
        except ValueError:
            # skip malformed lines
            continue
        rows.append(row)

    if not rows:
        arr = np.empty((0, 0), dtype=float)
    else:
        arr = np.array(rows, dtype=float)

        # try reshape if nx/ny present and total elements match
        if meta["nx"] is not None and meta["ny"] is not None:
            if arr.size == meta["nx"] * meta["ny"]:
                # VirtualLab seems to write rows as horizontal rows; reshape to (ny, nx)
                arr = arr.reshape(meta["ny"], meta["nx"])
            else:
                # if the file may have been written row by row but parsed as multiple rows,
                # attempt to coerce shape: if arr is (ny, nx) already do nothing
                pass

    return meta, comments_str, arr


# ----------------------------
# Combine two parsed files into complex ndarray
# meta1, arr1, meta2, arr2 expected
# Only accepts real+imag or imag+real or amp+phase or phase+amp
# ----------------------------
def combine_to_complex(meta1, arr1, meta2, arr2):
    t1 = meta1.get("data_type")
    t2 = meta2.get("data_type")

    if t1 is None or t2 is None:
        raise ValueError("data_type missing in one of the files; cannot combine.")

    if arr1.shape != arr2.shape:
        raise ValueError(f"Array shapes differ: {arr1.shape} vs {arr2.shape}")

    if (t1 == "real" and t2 == "imag"):
        return arr1 + 1j * arr2
    if (t1 == "imag" and t2 == "real"):
        return arr2 + 1j * arr1
    if (t1 == "amp" and t2 == "phase"):
        return arr1 * np.exp(1j * arr2)
    if (t1 == "phase" and t2 == "amp"):
        return arr2 * np.exp(1j * arr1)

    raise ValueError(f"Cannot combine data types: {t1} & {t2}")


# ----------------------------
# Visualization helpers (independent)
# - show_single: show a single array in one axes
# - show_side_by_side: show two arrays as subplots (single figure)
# - show_amplitude_phase: show amplitude and phase in subplots
# ----------------------------
def show_single(arr, title="", ax=None, cmap=None):
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax

def show_side_by_side(arr1, arr2, title1="", title2="", figsize=(10,5), cmap=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im1 = axes[0].imshow(arr1, origin="lower")
    axes[0].set_title(title1)
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(arr2, origin="lower")
    axes[1].set_title(title2)
    fig.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    return fig, axes

def show_amplitude_phase(cplx, title_amp="Amplitude", title_phase="Phase", figsize=(10,5)):
    amp = np.abs(cplx)
    ph = np.angle(cplx)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im0 = axes[0].imshow(amp, origin="lower")
    axes[0].set_title(title_amp)
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ph, origin="lower")
    axes[1].set_title(title_phase)
    fig.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    return fig, axes


# ----------------------------
# CLI main: keeps parsing and visualization independent
# usage:
#  python vlab_reader.py file.txt
#  python vlab_reader.py real.txt imag.txt
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="VirtualLab ASCII reader (parse + visualize)")
    parser.add_argument("files", nargs="+", help="One or two input .txt files")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful for scripts)")
    args = parser.parse_args()

    files = args.files
    if len(files) not in (1, 2):
        parser.error("Only 1 or 2 files supported.")

    if len(files) == 1:
        meta, comments, arr = parse_file(files[0])
        # print meta and comments
        print("meta:", meta)
        print("comments (first 500 chars):\n", comments[:500])
        # visualize single in one subplot
        show_single(arr, title=f"{files[0]} ({meta.get('data_type')})")
        if not args.no_show:
            plt.show()

    else:
        meta1, comments1, arr1 = parse_file(files[0])
        meta2, comments2, arr2 = parse_file(files[1])

        print("file1 meta:", meta1)
        print("file2 meta:", meta2)
        # show two raw maps side-by-side (single figure with 2 subplots)
        fig, axes = show_side_by_side(arr1, arr2,
                                      title1=f"{files[0]} ({meta1.get('data_type')})",
                                      title2=f"{files[1]} ({meta2.get('data_type')})")
        if not args.no_show:
            plt.show()

        # combine if possible and show amplitude/phase (also in subplots)
        try:
            c = combine_to_complex(meta1, arr1, meta2, arr2)
            fig2, axes2 = show_amplitude_phase(c, title_amp="Amplitude (combined)", title_phase="Phase (combined)")
            if not args.no_show:
                plt.show()
        except Exception as e:
            print("Could not combine to complex:", e)


if __name__ == "__main__":
    main()
