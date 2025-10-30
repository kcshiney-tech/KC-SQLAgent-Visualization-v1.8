# hierarchical_chart.py
"""
多层级柱状图渲染器 — 最终版
✅ 图表宽度自适应（100vw）
✅ 横坐标同一级高度统一
✅ 标签动态换行
✅ 柱子上方显示数值
✅ 筛选框悬浮不占用宽度
"""
import json
import uuid
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------- 数据标准化 ----------
def _normalize_data(viz_data: Dict[str, Any]) -> Dict[str, Any]:
    if "data" in viz_data and "datasets" in viz_data.get("data", {}):
        cfg = viz_data
        raw_data = cfg.get("raw_data")
        if raw_data:
            viz_data = raw_data
        else:
            viz_data = {
                "title": cfg.get("options", {}).get("plugins", {}).get("title", {}).get("text", cfg.get("title", "多层级柱状图")),
                "xLabel": cfg.get("options", {}).get("scales", {}).get("x", {}).get("title", {}).get("text", cfg.get("xLabel", "分类")),
                "yLabel": cfg.get("options", {}).get("scales", {}).get("y", {}).get("title", {}).get("text", cfg.get("yLabel", "数值")),
                "labels": cfg["data"]["labels"],
                "values": [
                    {"label": ds["label"], "data": ds["data"]}
                    for ds in cfg["data"]["datasets"]
                ]
            }
    raw = json.loads(json.dumps(viz_data))
    raw["title"] = raw.get("title") or "多层级柱状图"
    raw["yLabel"] = raw.get("yLabel") or "数值"
    raw["xLabel"] = raw.get("xLabel") or "分类"
    if "labels" not in raw or "values" not in raw:
        raise ValueError("viz_data 必须包含 labels 和 values")

    merged_values = {}
    for v in raw["values"]:
        label = v["label"]
        data = v["data"]
        if label in merged_values:
            old = merged_values[label]
            merged = [a if a != 0 else b for a, b in zip(old, data)]
            merged_values[label] = merged
        else:
            merged_values[label] = data
    raw["values"] = [{"label": label, "data": data} for label, data in merged_values.items()]
    return raw


# ---------- 处理结构 ----------
def _extract_innermost_as_series(labels, values):
    if not labels or all('.' not in lbl for lbl in labels):
        return {"is_innermost_series": False, "clean_labels": labels, "series_list": values}
    clean_to_indices = {}
    inners = []
    clean_labels = []
    for i, lbl in enumerate(labels):
        if '.' in lbl:
            clean, inner = lbl.rsplit('.', 1)
        else:
            clean, inner = lbl, ''
        clean = clean.strip()
        inner = inner.strip()
        if clean not in clean_to_indices:
            clean_to_indices[clean] = []
            clean_labels.append(clean)
        clean_to_indices[clean].append(i)
        inners.append(inner)
    unique_inn = sorted(set(inners), key=lambda x: x.lower())
    n_clean = len(clean_labels)
    n_series = len(unique_inn)
    series_data = [[0] * n_clean for _ in range(n_series)]
    inner_to_idx = {inner: idx for idx, inner in enumerate(unique_inn)}
    for v in values:
        data = v["data"]
        for clean, orig_indices in clean_to_indices.items():
            c_idx = clean_labels.index(clean)
            for orig_idx in orig_indices:
                if orig_idx >= len(data):
                    continue
                val = data[orig_idx]
                if val == 0:
                    continue
                inner = inners[orig_idx]
                if inner in inner_to_idx:
                    s_idx = inner_to_idx[inner]
                    if series_data[s_idx][c_idx] == 0:
                        series_data[s_idx][c_idx] = val
    series_list = [{"label": unique_inn[i], "data": series_data[i]} for i in range(n_series)]
    return {"is_innermost_series": True, "clean_labels": clean_labels, "series_list": series_list}


def dedupe_merge_first_non_zero(labels, values_per_week):
    map_ = {}
    uniq = []
    week_cnt = len(values_per_week)
    merged = [[] for _ in range(week_cnt)]
    for i, lbl in enumerate(labels):
        if lbl not in map_:
            idx = len(uniq)
            map_[lbl] = idx
            uniq.append(lbl)
            for w in range(week_cnt):
                merged[w].append(0)
        idx = map_[lbl]
        for w in range(week_cnt):
            v = values_per_week[w]["data"][i] if i < len(values_per_week[w]["data"]) else 0
            if merged[w][idx] == 0 and v != 0:
                merged[w][idx] = v
    for w in range(week_cnt):
        while len(merged[w]) < len(uniq):
            merged[w].append(0)
    weeks = [v["label"] for v in values_per_week]
    return {"uniqueLabels": uniq, "merged": merged, "weeks": weeks}


def split_rows(labels):
    rows = [s.split('.') for s in labels]
    max_d = max((len(r) for r in rows), default=0)
    for r in rows:
        r.extend([''] * (max_d - len(r)))
    return {"rows": rows, "maxDepth": max_d}


def hierarchical_sort(labels):
    split = split_rows(labels)
    rows, maxDepth = split["rows"], split["maxDepth"]
    idx = list(range(len(labels)))
    idx.sort(key=lambda i: tuple(rows[i]))
    return {"indices": idx, "rows": [rows[i] for i in idx], "maxDepth": maxDepth}


def build_hierarchy(rows):
    def rec(level, s, e):
        if level >= len(rows[0]):
            return []
        nodes = []
        i = s
        while i <= e:
            val = rows[i][level]
            j = i + 1
            while j <= e and rows[j][level] == val:
                j += 1
            node = {"start": i, "end": j - 1, "label": val, "children": []}
            node["children"] = rec(level + 1, i, j - 1)
            nodes.append(node)
            i = j
        return nodes
    return rec(0, 0, len(rows) - 1)


def collect_groups(root, maxDepth):
    groupsPerLevel = [[] for _ in range(maxDepth)]

    def traverse(node, level):
        if level < maxDepth:
            groupsPerLevel[level].append({
                "start": node["start"],
                "end": node["end"],
                "label": node["label"]
            })
        for child in node["children"]:
            traverse(child, level + 1)

    for child in root:
        traverse(child, 0)
    return groupsPerLevel


# ---------- 渲染 ----------
def render_hierarchical_bar(viz_data: Dict[str, Any], height: int = 700) -> Tuple[str, int]:
    chart_id = f"echarts_{uuid.uuid4().hex}"
    panel_id = f"legend_panel_{uuid.uuid4().hex[:8]}"

    try:
        raw = _normalize_data(viz_data)
        extracted = _extract_innermost_as_series(raw["labels"], raw["values"])
        if extracted["is_innermost_series"]:
            uniqLabels = extracted["clean_labels"]
            merged = [s["data"] for s in extracted["series_list"]]
            weeks = [s["label"] for s in extracted["series_list"]]
        else:
            deduped = dedupe_merge_first_non_zero(raw["labels"], raw["values"])
            uniqLabels = deduped["uniqueLabels"]
            merged = deduped["merged"]
            weeks = deduped["weeks"]

        sorted_info = hierarchical_sort(uniqLabels)
        order = sorted_info["indices"]
        sortedRows = sorted_info["rows"]
        maxDepth = sorted_info["maxDepth"]

        uniqLabels = [uniqLabels[i] for i in order]
        sortedMerged = [[row[i] for i in order] for row in merged]

        root = build_hierarchy(sortedRows)
        groupsPerLevel = collect_groups(root, maxDepth)
        splitAt = []

        labelFontSize = 12
        gapBetweenAxes = 36
        extraBottomPadding = 14
        gridBase = {"left": 70, "right": 40, "top": 80, "bottom": 80}
        min_height_per_level = 40
        maxLabelLines = 2
        reserved = maxDepth * min_height_per_level + extraBottomPadding + 60
        total_height = max(height, reserved + 120)
        grid = {**gridBase, "bottom": gridBase["bottom"] + reserved}

        palette = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#5ab1ef']

        html_template = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    html, body { margin:0; padding:0; height:100%; overflow:hidden; }
    .chart-wrapper { position:relative; width:100vw; left:50%; transform:translateX(-50%); height:__HEIGHT__px; }
    #__CHART_ID__ { width:100%; height:100%; }
    .__PANEL_ID__ {
      position:absolute; top:14px; right:24px; background:rgba(255,255,255,0.95);
      padding:10px 12px; border:1px solid #eee; border-radius:6px; box-shadow:0 4px 14px rgba(0,0,0,.08);
      z-index:9999; font-size:13px; min-width:160px; max-height:70vh; overflow:auto;
      backdrop-filter: blur(4px);
    }
  </style>
</head>
<body>
  <div class="chart-wrapper">
    <div id="__CHART_ID__"></div>
    <div class="__PANEL_ID__" id="__PANEL_ID__"></div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
  <script>
  function splitTextIntoLines(text, maxWidth, font, maxLines) {
    if (!text) return [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = font;
    if (ctx.measureText(text).width <= maxWidth) return [text];
    const words = text.split(' ');
    const lines = [];
    let current = words[0] || '';
    for (let i = 1; i < words.length; i++) {
      const test = current + ' ' + words[i];
      if (ctx.measureText(test).width <= maxWidth) current = test;
      else { lines.push(current); current = words[i]; }
    }
    lines.push(current);
    for (let i = 0; i < lines.length; i++) {
      if (ctx.measureText(lines[i]).width > maxWidth) {
        const s = lines[i];
        lines.splice(i, 1);
        let cur = '';
        for (let ch of s) {
          if (ctx.measureText(cur + ch).width <= maxWidth) cur += ch;
          else { if (cur) lines.push(cur); cur = ch; }
        }
        if (cur) lines.push(cur);
      }
    }
    if (lines.length > maxLines) {
      const truncated = lines.slice(0, maxLines);
      const ell = '…';
      let last = truncated[truncated.length - 1];
      while (ctx.measureText(last + ell).width > maxWidth && last.length > 0) last = last.slice(0, -1);
      truncated[truncated.length - 1] = last + ell;
      return truncated;
    }
    return lines;
  }

  (function() {
    const raw = __RAW_JSON__;
    const uniqLabels = __UNIQLABELS_JSON__;
    const sortedMerged = __SORTEDMERGED_JSON__;
    const weeks = __WEEKS_JSON__;
    const maxDepth = __MAXDEPTH__;
    const groupsPerLevel = __GROUPSPERLEVEL_JSON__;
    const grid = {left: __GRID_LEFT__, right: __GRID_RIGHT__, top: __GRID_TOP__, bottom: __GRID_BOTTOM__};
    const palette = __PALETTE_JSON__;
    const fontSpec = `${__LABELFONTSIZE__}px sans-serif`;
    const maxLines = __MAX_LABEL_LINES__;

    const chart = echarts.init(document.getElementById('__CHART_ID__'));
    const series = weeks.map((wk, i) => ({
      name: wk, type: 'bar', data: sortedMerged[i],
      itemStyle: {color: palette[i % palette.length]},
      barMaxWidth: 28,
      label: {
        show: true, position: 'top', fontSize: 11, color: '#333',
        formatter: p => p.value > 0 ? p.value : ''
      }
    }));

    chart.setOption({
      title: {text: raw.title, left: 'center'},
      tooltip: {trigger: 'axis', axisPointer: {type: 'shadow'}},
      legend: {show: false},
      grid, yAxis: {type: 'value', name: raw.yLabel},
      xAxis: {type: 'category', data: uniqLabels.map((_,i)=>i+''), axisLabel:{show:false}},
      series
    });

    // ✅ draw() 使用统一高度 + 动态换行
    function draw() {
      const w = chart.getWidth(), h = chart.getHeight();
      const {left, right, top, bottom} = grid;
      const innerW = w - left - right;
      const catW = innerW / Math.max(1, uniqLabels.length);
      const gfx = [];
      const bgColors = ['#f5f5f5','#f9f9f9','#fcfcfc','#fefefe'];
      const reservedTop = h - bottom;
      const ctx = document.createElement('canvas').getContext('2d');
      ctx.font = fontSpec;

      const levelHeights = new Array(maxDepth).fill(0);
      const levelLabelsLines = new Array(maxDepth).fill(null).map(()=>[]);

      for (let lvl = 0; lvl < maxDepth; lvl++) {
        const gs = groupsPerLevel[lvl] || [];
        for (const g of gs) {
          if (!g.label) continue;
          const wRect = (g.end - g.start + 1) * catW;
          const labelWidth = ctx.measureText(g.label).width;
          let lines = [];
          if (labelWidth <= wRect * 0.9) lines = [g.label];
          else lines = splitTextIntoLines(g.label, wRect * 0.9, fontSpec, maxLines);
          const lineHeight = __LABELFONTSIZE__ * 1.2;
          const totalTextHeight = lines.length * lineHeight;
          levelLabelsLines[lvl].push({g, lines, totalTextHeight});
          levelHeights[lvl] = Math.max(levelHeights[lvl], totalTextHeight + 10);
        }
        if (levelHeights[lvl] < __GAPBETWEENAXES__) levelHeights[lvl] = __GAPBETWEENAXES__;
      }

      let yCursor = reservedTop;
      for (let lvl = 0; lvl < maxDepth; lvl++) {
        const bg = bgColors[lvl % bgColors.length];
        const gs = levelLabelsLines[lvl];
        const levelH = levelHeights[lvl];
        for (const {g, lines, totalTextHeight} of gs) {
          const x = left + g.start * catW;
          const wRect = (g.end - g.start + 1) * catW;
          const y0 = yCursor + ((maxDepth - 1 - lvl) * levelH) + 6;
          gfx.push({type:'rect',shape:{x,y:y0,width:wRect,height:levelH},style:{fill:bg,stroke:'#e6e6e6',lineWidth:1},z:1});
          const lineHeight = __LABELFONTSIZE__ * 1.2;
          lines.forEach((line,i)=>{
            const textY = y0 + (levelH - totalTextHeight)/2 + lineHeight*(i+0.8);
            gfx.push({type:'text',style:{text:line,x:x+wRect/2,y:textY,textAlign:'center',textVerticalAlign:'middle',fill:'#222',font:fontSpec},z:2});
          });
        }
      }

      chart.setOption({graphic:gfx});
    }
    draw();
    window.addEventListener('resize',()=>{chart.resize();setTimeout(draw,120);});
  })();
  </script>
</body>
</html>"""

        replacements = {
            "__TITLE__": json.dumps(raw.get("title", "多层级柱状图"), ensure_ascii=False).strip('"'),
            "__HEIGHT__": str(total_height),
            "__CHART_ID__": chart_id,
            "__PANEL_ID__": panel_id,
            "__RAW_JSON__": json.dumps(raw, ensure_ascii=False),
            "__UNIQLABELS_JSON__": json.dumps(uniqLabels, ensure_ascii=False),
            "__SORTEDMERGED_JSON__": json.dumps(sortedMerged, ensure_ascii=False),
            "__WEEKS_JSON__": json.dumps(weeks, ensure_ascii=False),
            "__MAXDEPTH__": str(maxDepth),
            "__GROUPSPERLEVEL_JSON__": json.dumps(groupsPerLevel, ensure_ascii=False),
            "__PALETTE_JSON__": json.dumps(palette, ensure_ascii=False),
            "__GRID_LEFT__": str(grid["left"]),
            "__GRID_RIGHT__": str(grid["right"]),
            "__GRID_TOP__": str(grid["top"]),
            "__GRID_BOTTOM__": str(grid["bottom"]),
            "__LABELFONTSIZE__": str(labelFontSize),
            "__GAPBETWEENAXES__": str(gapBetweenAxes),
            "__MAX_LABEL_LINES__": str(maxLabelLines),
        }
        html = html_template
        for k, v in replacements.items():
            html = html.replace(k, v)
        return html, total_height

    except Exception as e:
        logger.error(f"hierarchical_chart 渲染失败: {e}\n输入: {json.dumps(viz_data, ensure_ascii=False)[:1000]}")
        err_html = f"<div style='padding:20px;color:red;'>图表渲染失败: {str(e)[:200]}</div>"
        return err_html, height


__all__ = ["render_hierarchical_bar"]
