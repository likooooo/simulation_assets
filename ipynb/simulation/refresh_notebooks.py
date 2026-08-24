#!/usr/bin/env python3
"""Re-execute ipynb cells and embed Agg matplotlib output as notebook PNGs."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("SAVE_TO_FILE", "0")

try:
    from py_core_plugins import viz_io
except ImportError:
    import viz_io

viz_io.sync_save_to_file_from_env()

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt


def _stamp_notebook_refresh_markdown(nb: dict[str, Any], ts: str) -> None:
    stamp = f"**刷新时间:** `{ts}`\n"
    cells = nb.setdefault("cells", [])
    md_idx = next((i for i, c in enumerate(cells) if c.get("cell_type") == "markdown"), None)
    if md_idx is None:
        cells.insert(
            0,
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": [stamp + "\n"],
            },
        )
        return
    cell = cells[md_idx]
    src = cell.get("source", [])
    text = "".join(src) if isinstance(src, list) else str(src)
    lines = text.splitlines(keepends=True)
    lines = [ln for ln in lines if not ln.startswith("**刷新时间:**")]
    if lines and lines[0].strip() == "":
        lines = lines[1:]
    cell["source"] = [stamp + "\n", *lines]


def _figure_to_output(fig) -> dict[str, Any]:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {"image/png": b64, "text/plain": "<IPython.core.display.Image object>"},
    }


def refresh_notebook(nb_path: Path) -> int:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    root = nb_path.parent.resolve()
    print(f"refreshing {nb_path} at {ts}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    _stamp_notebook_refresh_markdown(nb, ts)
    for cell in nb.get("cells", []):
        if "id" not in cell:
            cell["id"] = uuid.uuid4().hex[:8]
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    ns: dict[str, Any] = {"__name__": "__main__", "__file__": str(nb_path)}
    exec_count = 0
    cwd_prev = Path.cwd()
    first_code = True
    try:
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            exec_count += 1
            cell["execution_count"] = exec_count
            src = cell.get("source", [])
            code = "".join(src) if isinstance(src, list) else str(src)
            code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("%"))
            if first_code:
                stamp_line = f'print("notebook refreshed at {ts}")'
                lines = code.splitlines()
                insert_at = 0
                while insert_at < len(lines) and (
                    not lines[insert_at].strip() or lines[insert_at].strip().startswith("from __future__")
                ):
                    insert_at += 1
                lines.insert(insert_at, stamp_line)
                code = "\n".join(lines)
                first_code = False
            outputs: list[dict[str, Any]] = []
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            def _flush_stream(buf: io.StringIO, name: str) -> None:
                text = buf.getvalue()
                if not text:
                    return
                buf.seek(0)
                buf.truncate(0)
                outputs.append({"output_type": "stream", "name": name, "text": text})

            def _capture_figure(fig=None):
                _flush_stream(stdout_buf, "stdout")
                _flush_stream(stderr_buf, "stderr")
                if fig is None:
                    fig = plt.gcf()
                outputs.append(_figure_to_output(fig))

            old_viz_show = viz_io.show_figure
            old_plt_show = plt.show
            viz_io.show_figure = _capture_figure  # type: ignore[assignment]
            plt.show = _capture_figure  # type: ignore[assignment]
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    try:
                        exec(compile(code, str(nb_path), "exec"), ns, ns)
                    except SystemExit as exc:
                        code_rc = int(getattr(exc, "code", 0) or 0)
                        if code_rc != 0:
                            raise RuntimeError(f"{nb_path.name}: cell exit code {code_rc}") from exc
                    except Exception as exc:
                        _flush_stream(stdout_buf, "stdout")
                        _flush_stream(stderr_buf, "stderr")
                        outputs.append(
                            {
                                "output_type": "error",
                                "ename": type(exc).__name__,
                                "evalue": str(exc),
                                "traceback": [str(exc)],
                            }
                        )
                        cell["outputs"] = outputs
                        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                        raise
            finally:
                viz_io.show_figure = old_viz_show  # type: ignore[assignment]
                plt.show = old_plt_show
            _flush_stream(stdout_buf, "stdout")
            _flush_stream(stderr_buf, "stderr")
            cell["outputs"] = outputs
    finally:
        os.chdir(cwd_prev)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_png = sum(
        1
        for c in nb.get("cells", [])
        for o in c.get("outputs", [])
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})
    )
    print(f"refreshed {nb_path.name}: {n_png} figure(s)")
    return n_png


def refresh_notebooks(paths: Sequence[str | os.PathLike[str]]) -> int:
    batch_ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"notebook refresh start: {batch_ts}")
    os.environ["SAVE_TO_FILE"] = "0"
    viz_io.sync_save_to_file_from_env()
    failed = 0
    for raw in paths:
        try:
            refresh_notebook(Path(raw).resolve())
        except Exception as exc:
            failed += 1
            print(f"FAILED {Path(raw).name}: {exc}", file=sys.stderr)
    return failed


def _git_modified_ipynb(assets_root: Path) -> list[Path]:
    import subprocess

    out = subprocess.check_output(
        ["git", "-C", str(assets_root), "diff", "--name-only", "HEAD"],
        text=True,
    )
    unstaged = subprocess.check_output(
        ["git", "-C", str(assets_root), "diff", "--name-only"],
        text=True,
    )
    names = {ln.strip() for ln in (out + unstaged).splitlines() if ln.strip().endswith(".ipynb")}
    return sorted(assets_root / n for n in names)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    assets_root = Path(__file__).resolve().parents[2]
    if not argv:
        paths = _git_modified_ipynb(assets_root)
        if not paths:
            print("refresh_notebooks: no modified ipynb under assets")
            return 0
    else:
        paths = [Path(p) if Path(p).is_absolute() else assets_root / p for p in argv]
    return refresh_notebooks(paths)


if __name__ == "__main__":
    raise SystemExit(main())
