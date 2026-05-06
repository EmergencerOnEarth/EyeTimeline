#!/usr/bin/env python3
"""
推理结果 Web 浏览服务（树状目录 + 每样本三张图）

与 infer_mae.py 输出约定一致：子目录内包含
  original.png / masked.png / reconstructed.png
即视为一个「样本」节点；其余子目录为可展开文件夹。

用法：
  pip install flask
  python infer_viewer_server.py --root /path/to/infer_output --port 8765

浏览器打开 http://<host>:8765/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

SAMPLE_FILES = ("original.png", "masked.png", "reconstructed.png")


def is_sample_dir(d: Path) -> bool:
    return d.is_dir() and all((d / name).is_file() for name in SAMPLE_FILES)


def resolve_under_root(root: Path, *rel_parts: str) -> Path:
    """将相对路径段解析为绝对路径，且必须落在 root 之下。"""
    root = root.resolve()
    if not rel_parts or rel_parts == ("",):
        return root
    candidate = (root / Path(*rel_parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        abort(403)
    return candidate


def create_app(root: Path) -> Flask:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"根目录不存在或不是文件夹: {root}")

    app = Flask(__name__)

    @app.route("/")
    def index():
        from flask import render_template_string

        return render_template_string(INDEX_HTML, root_display=str(root))

    @app.route("/api/browse")
    def api_browse():
        rel = request.args.get("path", "") or ""
        rel = rel.replace("\\", "/").strip("/")
        parts = [p for p in rel.split("/") if p and p != "."]
        if any(p == ".." for p in parts):
            abort(403)
        target = resolve_under_root(root, *parts)
        if not target.is_dir():
            return jsonify(error="not a directory"), 400

        # 直接把 --root（或展开的子路径）指向「样本目录」时，无子节点可列，仅展示三图。
        if is_sample_dir(target):
            return jsonify(
                root=str(root),
                cwd=rel,
                entries=[],
                is_sample_here=True,
                sample_rel_path=rel.replace("\\", "/"),
            )

        entries = []
        try:
            children = sorted(
                target.iterdir(),
                key=lambda p: (
                    0 if p.is_dir() else 1,
                    p.name.lower(),
                ),
            )
        except OSError as e:
            return jsonify(error=str(e)), 500

        for child in children:
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            rel_p = str(child.relative_to(root)).replace(os.sep, "/")
            if is_sample_dir(child):
                entries.append(
                    {
                        "name": child.name,
                        "path": rel_p,
                        "kind": "sample",
                    }
                )
            else:
                entries.append(
                    {
                        "name": child.name,
                        "path": rel_p,
                        "kind": "folder",
                    }
                )

        return jsonify(root=str(root), cwd=rel, entries=entries)

    @app.route("/raw/<path:rel_path>")
    def raw_file(rel_path: str):
        rel_path = rel_path.replace("\\", "/").strip("/")
        parts = [p for p in rel_path.split("/") if p and p != "."]
        if any(p == ".." for p in parts):
            abort(403)
        full = resolve_under_root(root, *parts)
        if not full.is_file():
            abort(404)
        return send_file(full)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="推理结果目录 Web 浏览")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("infer_output"),
        help="推理输出根目录（默认 ./infer_output）",
    )
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=8765, help="端口")
    p.add_argument("--debug", action="store_true", help="Flask debug（仅本机调试用）")
    args = p.parse_args()

    app = create_app(args.root)
    print(f"[infer_viewer] root = {args.root.resolve()}")
    print(f"[infer_viewer] http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>推理结果浏览 — {{ root_display }}</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --border: #2d3a4d;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --accent: #58a6ff;
      --sample: #3fb950;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
      display: flex; flex-direction: column;
    }
    header {
      padding: 0.75rem 1.25rem; border-bottom: 1px solid var(--border);
      background: var(--panel); font-size: 0.9rem;
    }
    header code { color: var(--accent); word-break: break-all; }
    main { display: flex; flex: 1; min-height: 0; }
    #tree-panel {
      width: min(380px, 40vw); border-right: 1px solid var(--border);
      overflow: auto; padding: 0.5rem 0; background: #121820;
    }
    #detail {
      flex: 1; overflow: auto; padding: 1rem 1.25rem;
    }
    .tree-item {
      user-select: none; padding: 0.2rem 0.5rem 0.2rem 0.25rem;
      cursor: pointer; border-radius: 4px; margin: 0 0.35rem;
      display: flex; align-items: center; gap: 0.35rem; font-size: 0.875rem;
    }
    .tree-item:hover { background: rgba(88, 166, 255, 0.12); }
    .tree-item.sample { color: var(--sample); }
    .tree-item.sample::before { content: "◆"; font-size: 0.65rem; opacity: 0.85; }
    .tree-item.folder::before { content: "▸"; display: inline-block; transition: transform 0.15s; width: 0.9em; }
    .tree-item.folder.open::before { transform: rotate(90deg); }
    .children { margin-left: 1rem; border-left: 1px solid var(--border); padding-left: 0.25rem; }
    .hint { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
    .imgs {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem; margin-top: 1rem;
    }
    .img-card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
      overflow: hidden;
    }
    .img-card h3 {
      margin: 0; padding: 0.5rem 0.75rem; font-size: 0.8rem; font-weight: 600;
      color: var(--muted); border-bottom: 1px solid var(--border);
    }
    .img-card img {
      width: 100%; height: auto; display: block; background: #000;
    }
    .empty { color: var(--muted); padding: 2rem; text-align: center; }
    #load-err { color: #f85149; font-size: 0.85rem; margin: 0.5rem 1rem; }
  </style>
</head>
<body>
  <header>
    根目录 <code>{{ root_display }}</code>
  </header>
  <main>
    <div id="tree-panel">
      <div id="load-err"></div>
      <div id="tree-root" class="children"></div>
    </div>
    <div id="detail">
      <p class="hint">左侧点击文件夹展开；点击带 ◆ 的样本查看 original / masked / reconstructed 三图。</p>
      <div id="sample-view"></div>
    </div>
  </main>
  <script>
    const enc = (p) => p.split("/").map(encodeURIComponent).join("/");

    async function fetchBrowse(relPath) {
      const q = relPath ? "?path=" + encodeURIComponent(relPath) : "";
      const r = await fetch("/api/browse" + q);
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || r.statusText);
      }
      return r.json();
    }

    function renderSample(path) {
      const base = path ? "/raw/" + enc(path) : "/raw";
      document.getElementById("sample-view").innerHTML = `
        <h2 style="font-size:1rem;font-weight:600;margin:0 0 0.5rem 0;">${path}</h2>
        <div class="imgs">
          <div class="img-card"><h3>original</h3><img src="${base}/original.png" alt="original" /></div>
          <div class="img-card"><h3>masked</h3><img src="${base}/masked.png" alt="masked" /></div>
          <div class="img-card"><h3>reconstructed</h3><img src="${base}/reconstructed.png" alt="reconstructed" /></div>
        </div>`;
    }

    async function expandFolder(el, path) {
      const childWrap = el._childWrap;
      if (el._loaded) {
        el.classList.toggle("open");
        childWrap.style.display = el.classList.contains("open") ? "block" : "none";
        return;
      }
      const data = await fetchBrowse(path);
      el._loaded = true;
      el.classList.add("open");
      childWrap.style.display = "block";
      for (const e of data.entries) {
        childWrap.appendChild(await buildNode(e, path));
      }
      if (data.entries.length === 0) {
        const empty = document.createElement("div");
        empty.className = "hint";
        empty.style.marginLeft = "1rem";
        empty.textContent = "(空目录)";
        childWrap.appendChild(empty);
      }
    }

    async function buildNode(entry, parentPath) {
      const row = document.createElement("div");
      if (entry.kind === "sample") {
        row.className = "tree-item sample";
        row.textContent = entry.name;
        row.onclick = () => renderSample(entry.path);
        return row;
      }
      const folder = document.createElement("div");
      const head = document.createElement("div");
      head.className = "tree-item folder";
      head.textContent = entry.name;
      const childWrap = document.createElement("div");
      childWrap.className = "children";
      childWrap.style.display = "none";
      head._childWrap = childWrap;
      head.onclick = () => {
        expandFolder(head, entry.path).catch((err) => {
          document.getElementById("load-err").textContent = err.message;
        });
      };
      folder.appendChild(head);
      folder.appendChild(childWrap);
      return folder;
    }

    (async () => {
      const rootEl = document.getElementById("tree-root");
      try {
        const data = await fetchBrowse("");
        if (data.is_sample_here) {
          renderSample(data.sample_rel_path || "");
          rootEl.innerHTML =
            '<div class="hint" style="margin:0.75rem 1rem;">当前根目录即样本文件夹（三张图已在右侧）。</div>';
          return;
        }
        if (data.entries.length === 0) {
          rootEl.innerHTML = '<div class="empty">根目录下没有子文件夹。</div>';
          return;
        }
        for (const e of data.entries) {
          rootEl.appendChild(await buildNode(e, ""));
        }
      } catch (err) {
        document.getElementById("load-err").textContent = err.message;
      }
    })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
