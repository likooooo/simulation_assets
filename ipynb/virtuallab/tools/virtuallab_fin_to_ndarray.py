#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VirtualLab Complex Field Reader (Updated)
Supports formats:
1. "Re - iIm" or "Re + iIm"
2. "Amp · exp(Phase · i)" (from 2.txt)
"""

import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import cmath

# ----------------------------
# Helper: Convert VLab string to python complex
# ----------------------------
def parse_vlab_complex_string(token):
    token = token.strip()
    if not token or token == '0':
        return 0j
    
    # --- Case 1: Amplitude * exp(Phase * i) ---
    # Example: "5.75E-14 · exp(-0.997 · i)"
    if 'exp(' in token:
        try:
            # Split into Amp and the rest
            # Regex to handle various dots (·) or stars (*) and spaces
            # Matches: (Amplitude) [separator] exp( (Phase) [separator] i)
            match = re.match(r"([+\-0-9.Ee]+)\s*[\·\*]\s*exp\(\s*([+\-0-9.Ee]+)\s*[\·\*]\s*i\s*\)", token)
            if match:
                amp = float(match.group(1))
                phase = float(match.group(2))
                return cmath.rect(amp, phase)
        except Exception:
            pass # Fallback to other parsers if regex fails

    # --- Case 2: Real +/- i*Imag ---
    # Example: "3.11E-14 - i4.83E-14" or "3.11E-14 + i4.83E-14"
    # Remove whitespace to make splitting consistent
    # Note: simple split might fail if Real part is negative, but usually VLab puts space around operator
    
    # Try handling " - i"
    if ' - i' in token:
        parts = token.split(' - i')
        if len(parts) == 2:
            try:
                return complex(float(parts[0]), -float(parts[1]))
            except: pass

    # Try handling " + i"
    if ' + i' in token:
        parts = token.split(' + i')
        if len(parts) == 2:
            try:
                return complex(float(parts[0]), float(parts[1]))
            except: pass
            
    # --- Case 3: Fallback (Real only) ---
    try:
        return complex(float(token))
    except ValueError:
        return 0j

# ----------------------------
# Parse file
# ----------------------------
def parse_file(path):
    meta = {
        "nx": None, "ny": None, 
        "dx": None, "dy": None, 
        "wavelength": None
    }
    
    raw_data_tokens = []
    
    # Use latin-1 or utf-8. 2.txt might contain special dot characters.
    # 'utf-8' usually handles the special dot '·' fine.
    with open(path, "r", encoding="utf-8", errors='replace') as f:
        for line in f:
            line = line.rstrip("\n")
            
            if line.startswith("#") or not line.strip():
                # Parse Header Info
                
                # Nx, Ny
                m_n = re.search(r"Number of Data Points\s*:\s*\(\s*(\d+)\s*[;,]\s*(\d+)\s*\)", line, re.IGNORECASE)
                if m_n:
                    meta["nx"] = int(m_n.group(1))
                    meta["ny"] = int(m_n.group(2))

                # dx, dy
                m_d = re.search(r"Sampling Distance.*:\s*\(\s*([0-9Ee\.\+\-]+)\s*[;,]\s*([0-9Ee\.\+\-]+)\s*\)", line, re.IGNORECASE)
                if m_d:
                    meta["dx"] = float(m_d.group(1))
                    meta["dy"] = float(m_d.group(2))
                    
                # Wavelength
                m_wl = re.search(r"Wavelength.*:\s*([0-9Ee\.\+\-]+)", line, re.IGNORECASE)
                if m_wl:
                    meta["wavelength"] = float(m_wl.group(1))
            else:
                # Data lines: split by tab or multiple spaces
                tokens = re.split(r"\t+", line.strip())
                tokens = [t for t in tokens if t]
                raw_data_tokens.extend(tokens)

    # Parse numeric data
    # Using a loop to handle potential errors gracefully
    complex_data = []
    for t in raw_data_tokens:
        complex_data.append(parse_vlab_complex_string(t))
    
    arr = np.array(complex_data, dtype=complex)
    
    # Reshape
    if meta["nx"] and meta["ny"]:
        if arr.size == meta["nx"] * meta["ny"]:
            arr = arr.reshape(meta["ny"], meta["nx"])
        else:
            print(f"Warning: Data count ({arr.size}) != Nx*Ny ({meta['nx']}*{meta['ny']})")
            
    return meta, arr

# ----------------------------
# Visualization: 2x2 Subplot
# ----------------------------
def show_complex_plot(cplx_arr, meta, title_prefix="", pos=[[0,0], [0,1], [1, 0], [1, 1]]):
    # Physical coordinates if available
    extent = None
    xlabel, ylabel = "x (px)", "y (px)"
    
    if meta["dx"] and meta["dy"] and meta["nx"] and meta["ny"]:
        width = meta["nx"] * meta["dx"]
        height = meta["ny"] * meta["dy"]
        # Assuming centered field usually, but 0 to W/H is safer default
        extent = [0, width, 0, height]
        xlabel, ylabel = "x (mm)", "y (mm)"

    # Calculate components
    real_part = np.real(cplx_arr)
    imag_part = np.imag(cplx_arr)
    amplitude = np.abs(cplx_arr)
    phase = np.angle(cplx_arr)

    # Create 2x2 plot
    max_x = np.max([x for x,y in pos])
    max_y = np.max([y for x,y in pos])
    fig, axes = plt.subplots(max_x + 1, max_y + 1, figsize=(12, 10))
    fig.suptitle(f"Field Analysis: {title_prefix}", fontsize=16)

    is_1d_layout = False    
    if max_x == 0:
        is_1d_layout = True
        pos = [y for x,y in pos]
    if max_y == 0:
        is_1d_layout = True
        pos = [x for x,y in pos]

    # 1. Real Part (Top Left)
    ax = axes[pos[0]] if is_1d_layout  else axes[*pos[0]]
    im = ax.imshow(real_part, origin="lower", extent=extent, cmap='RdBu_r') # Diverging cmap for +/-
    ax.set_title("Real Part")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    # 2. Imaginary Part (Top Right)
    ax = axes[pos[1]] if is_1d_layout  else axes[*pos[1]]
    im = ax.imshow(imag_part, origin="lower", extent=extent, cmap='RdBu_r')
    ax.set_title("Imaginary Part")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    # 3. Amplitude (Bottom Left)
    ax = axes[pos[2]] if is_1d_layout  else axes[*pos[2]]
    im = ax.imshow(amplitude, origin="lower", extent=extent, cmap='inferno')
    ax.set_title("Amplitude")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    # 4. Phase (Bottom Right)
    ax = axes[pos[3]] if is_1d_layout  else axes[*pos[3]]
    im = ax.imshow(phase, origin="lower", extent=extent, cmap='twilight') # Cyclic cmap for phase
    ax.set_title("Phase (rad)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="VirtualLab Complex Field Reader (Universal)")
    parser.add_argument("file", help="Path to the .txt file")
    parser.add_argument("--no-show", action="store_true", help="Do not show plot window")
    
    args = parser.parse_args()
    
    print(f"Reading file: {args.file}...")
    try:
        meta, arr = parse_file(args.file)
        
        print("-" * 30)
        print("Metadata Found:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"Array Shape: {arr.shape}")
        print("-" * 30)

        if arr.size > 0:
            show_complex_plot(arr, meta, title_prefix=args.file)
            if not args.no_show:
                plt.show()
        else:
            print("Error: No data found in file.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()