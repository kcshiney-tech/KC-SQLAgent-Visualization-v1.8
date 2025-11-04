# hierarchical_chart.py
"""
多层级柱状图渲染器 — 最终整合版
- 兼容时间序列 input、最里层子标签 input 与回退去重合并策略
- 动态换行（按每个组宽度判断）、同级统一高度
- 柱顶数值显示（可开关）
- 筛选面板悬浮右上角（全选/取消/显示数值）
- 每次更新 series 后重绘 graphic（避免被 chart.setOption 清空）
"""
from typing import Dict, Any, Tuple, List, Optional
import json
import uuid
import re
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathlib 
# 读取本地文件（选用绝对/相对路径，确保路径正确）
base = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "static" / "js"

echarts_js = (base / "echarts.min.js").read_text(encoding="utf-8")

logger = logging.getLogger(__name__)


# -------------------------
# Helpers: data normalization and transformations
# -------------------------
def _normalize_data(viz_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容输入：
    - Chart.js 风格完整 config: { type, data: { labels, datasets }, options }
    - 内部简化格式: { labels: [...], values: [ {label:..., data:[...]}, ... ], title, xLabel, yLabel }
    返回规范化结构（raw）包含 keys: title, xLabel, yLabel, labels, values
    """
    if isinstance(viz_data, dict) and "data" in viz_data and isinstance(viz_data.get("data"), dict) and "datasets" in viz_data["data"]:
        cfg = viz_data
        raw_data = cfg.get("raw_data")
        if raw_data:
            viz_data = raw_data
        else:
            viz_data = {
                "title": cfg.get("options", {}).get("plugins", {}).get("title", {}).get("text", cfg.get("title", "多层级柱状图")),
                "xLabel": cfg.get("options", {}).get("scales", {}).get("x", {}).get("title", {}).get("text", cfg.get("xLabel", "分类")),
                "yLabel": cfg.get("options", {}).get("scales", {}).get("y", {}).get("title", {}).get("text", cfg.get("yLabel", "数值")),
                "labels": cfg["data"].get("labels", []),
                "values": [
                    {"label": ds.get("label", f"series_{i}"), "data": ds.get("data", [])}
                    for i, ds in enumerate(cfg["data"].get("datasets", []))
                ]
            }
    raw = json.loads(json.dumps(viz_data))
    raw["title"] = raw.get("title") or "多层级柱状图"
    raw["yLabel"] = raw.get("yLabel") or "数值"
    raw["xLabel"] = raw.get("xLabel") or "分类"
    if "labels" not in raw or "values" not in raw:
        raise ValueError("viz_data 必须包含 labels 和 values")
    # 合并重复 label 的 values（保守合并：保留第一个非0）
    merged_values = {}
    for v in raw["values"]:
        label = v.get("label", "")
        data = v.get("data", [])
        if label in merged_values:
            old = merged_values[label]
            merged = [a if a != 0 else b for a, b in zip(old, data)]
            merged_values[label] = merged
        else:
            merged_values[label] = data
    raw["values"] = [{"label": label, "data": data} for label, data in merged_values.items()]
    return raw


def _looks_like_time_label(s: str) -> bool:
    """
    简单判断字符串是否像 'YYYY-MM' 或 'YYYY-M' 或 'YYYY-MM-DD' 这样的时间点标签
    """
    if not isinstance(s, str):
        return False
    return bool(re.match(r'^\d{4}(-\d{1,2})(-\d{1,2})?$', s))


def _detect_week_in_label(labels: List[str]) -> Tuple[bool, List[str], List[Optional[int]]]:
    """
    检测 labels 每个字符串末尾是否包含“周/周次/年月周”样式的标记。
    支持示例（末尾，以 '.' 分隔）：
      - ".2025-41"     -> 年-周 格式（提取 41）
      - ".第41周"      -> 中文“第N周”
      - ".W41" ".w41" ".week41" ".Week41" -> 英文周标记
      - ".41"          -> 直接数字（谨慎匹配）
    返回:
      - all_have_week: bool
      - clean_labels: labels 去掉末尾 week 标识的清理后版本
      - week_numbers: 每个 label 提取到的周数字（无法提取则为 None）
    """
    if not isinstance(labels, list):
        return False, labels or [], [None] * (len(labels or []))

    patterns = [
        r'\.(\d{4}-\d{1,2})$',            # .2025-41 或 .2025-5
        r'\.(第\d{1,2}周)$',              # .第41周
        r'\.(?:w|week|W|Week)?(\d{1,2})$',# .W41 / .week41 / .41 (捕获仅数字部分)
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    clean_labels: List[str] = []
    week_numbers: List[Optional[int]] = []

    for lbl in labels:
        if not isinstance(lbl, str):
            clean_labels.append(lbl)
            week_numbers.append(None)
            continue

        matched = False
        for pat in compiled:
            m = pat.search(lbl)
            if not m:
                continue
            group_vals = [g for g in m.groups() if g is not None]
            num_str = group_vals[-1] if group_vals else m.group(0)
            digits = re.findall(r'\d+', num_str)
            week_num = int(digits[-1]) if digits else None
            clean_labels.append(lbl[:m.start()])
            week_numbers.append(week_num)
            matched = True
            break

        if not matched:
            clean_labels.append(lbl)
            week_numbers.append(None)

    all_have_week = all(w is not None for w in week_numbers)
    return all_have_week, clean_labels, week_numbers


def _extract_innermost_as_series(labels: List[str], values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    如果 labels 中包含 '.' 分层，则将最后一段 (inner) 当作 series；
    返回 dict: { is_innermost_series: bool, clean_labels: [...], series_list: [ {label, data}, ... ] }
    """
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
        data = v.get("data", [])
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
                    # 保持合并策略：第一个非0优先
                    if series_data[s_idx][c_idx] == 0:
                        series_data[s_idx][c_idx] = val
    series_list = [{"label": unique_inn[i], "data": series_data[i]} for i in range(n_series)]
    return {"is_innermost_series": True, "clean_labels": clean_labels, "series_list": series_list}


def dedupe_merge_first_non_zero(labels: List[str], values_per_week: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    兜底合并策略：将相同 labels 合并到唯一列表上，取每列第一个非 0 值
    返回 { uniqueLabels, merged, weeks }
    """
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
            v = values_per_week[w].get("data", [])
            val = v[i] if i < len(v) else 0
            if merged[w][idx] == 0 and val != 0:
                merged[w][idx] = val
    for w in range(week_cnt):
        while len(merged[w]) < len(uniq):
            merged[w].append(0)
    weeks = [v.get("label", f"series_{i}") for i, v in enumerate(values_per_week)]
    return {"uniqueLabels": uniq, "merged": merged, "weeks": weeks}


def split_rows(labels: List[str]) -> Dict[str, Any]:
    rows = [s.split('.') for s in labels]
    max_d = max((len(r) for r in rows), default=0)
    for r in rows:
        r.extend([''] * (max_d - len(r)))
    return {"rows": rows, "maxDepth": max_d}


def hierarchical_sort(labels: List[str]) -> Dict[str, Any]:
    split = split_rows(labels)
    rows, maxDepth = split["rows"], split["maxDepth"]
    idx = list(range(len(labels)))
    idx.sort(key=lambda i: tuple(rows[i]))
    return {"indices": idx, "rows": [rows[i] for i in idx], "maxDepth": maxDepth}


def build_hierarchy(rows: List[List[str]]):
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


# -------------------------
# Main render function
# -------------------------
def render_hierarchical_bar(viz_data: Dict[str, Any], height: int = 700) -> Tuple[str, int]:
    """
    返回 (html_string, height)
    """
    chart_id = f"echarts_{uuid.uuid4().hex}"
    panel_id = f"legend_panel_{uuid.uuid4().hex[:8]}"

    try:
        raw = _normalize_data(viz_data)

        # --- 决定 series 来源：优先 time-like (values[].label), 然后 week-in-label, 再 innermost, 最后 dedupe 回退 ---
        values_list = raw.get("values", [])
        uniqLabels: List[str] = []
        merged = []
        weeks: List[str] = []

        # 1) 如果 values 是结构化的 series 且 values[].label 看起来像时间（YYYY-MM 等），直接使用
        if isinstance(values_list, list) and len(values_list) > 0 and all(isinstance(v, dict) and "label" in v for v in values_list):
            if all(_looks_like_time_label(v.get("label", "")) for v in values_list):
                uniqLabels = raw.get("labels", [])
                merged = [v.get("data", []) for v in values_list]
                weeks = [v.get("label", "") for v in values_list]
            else:
                # 2) 检测 raw.labels 中是否为 week-in-label（如 label: "xxx.2025-41"）
                all_week, clean_labels, week_numbers = _detect_week_in_label(raw.get("labels", []))
                if all_week:
                    # 使用 clean_labels 作为 x 轴（去掉末尾周标识），values 则仍作为 series（values[].label 作为 legend）
                    uniqLabels = clean_labels
                    merged = [v.get("data", []) for v in values_list]
                    weeks = [v.get("label", "") for v in values_list]
                else:
                    # 3) 否则尝试最里层提取（inner as series）
                    extracted = _extract_innermost_as_series(raw.get("labels", []), raw.get("values", []))
                    if extracted.get("is_innermost_series"):
                        uniqLabels = extracted.get("clean_labels", [])
                        merged = [s.get("data", []) for s in extracted.get("series_list", [])]
                        weeks = [s.get("label", "") for s in extracted.get("series_list", [])]
                    else:
                        # 4) 回退合并策略
                        deduped = dedupe_merge_first_non_zero(raw.get("labels", []), raw.get("values", []))
                        uniqLabels = deduped.get("uniqueLabels", [])
                        merged = deduped.get("merged", [])
                        weeks = deduped.get("weeks", [])
        else:
            # 兜底
            deduped = dedupe_merge_first_non_zero(raw.get("labels", []), raw.get("values", []))
            uniqLabels = deduped.get("uniqueLabels", [])
            merged = deduped.get("merged", [])
            weeks = deduped.get("weeks", [])

        # --- 确保排序与结构 ---
        sorted_info = hierarchical_sort(uniqLabels)
        order = sorted_info["indices"]
        sortedRows = sorted_info["rows"]
        maxDepth = sorted_info["maxDepth"]
        maxDepth = max(maxDepth, 1)

        uniqLabels = [uniqLabels[i] for i in order]
        # merged: list of series arrays -> reorder每个 series 的列顺序
        sortedMerged = [[row[i] for i in order] for row in merged]

        root = build_hierarchy(sortedRows)
        groupsPerLevel = collect_groups(root, maxDepth)

        # 布局参数
        labelFontSize = 12
        gapBetweenAxes = 36
        extraBottomPadding = 14
        gridBase = {"left": 70, "right": 40, "top": 80, "bottom": 80}
        min_height_per_level = 40
        maxLabelLines = 4
        reserved = maxDepth * min_height_per_level + extraBottomPadding + 60
        total_height = max(height, reserved + 120)
        grid = {**gridBase, "bottom": gridBase["bottom"] + reserved}

        palette = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#5ab1ef']

        # HTML 模板（使用占位符替换，避免 {} 冲突）
        # ------------------ 自动内联本地 JS（避免 /static 被 Streamlit 占用问题） ------------------
        import pathlib
        _here = pathlib.Path(__file__).resolve()

        # 尝试若干候选目录，自动选择第一个存在的 js 目录
        _candidate_dirs = [
            _here.parent / "static" / "js",                       # if hierarchy.py sits in frontend/
            _here.parents[1] / "frontend" / "static" / "js",      # if hierarchy.py is one level below project root
            _here.parents[1] / "static" / "js",                   # alternate
            _here.parents[2] / "frontend" / "static" / "js",      # deeper layout
            _here.parents[2] / "static" / "js",
            pathlib.Path.cwd() / "frontend" / "static" / "js",    # run-cwd relative
            pathlib.Path.cwd() / "static" / "js"
        ]
        js_dir = None
        for d in _candidate_dirs:
            if d.exists() and d.is_dir():
                js_dir = d
                break

        def _read_js_or_warn(fname):
            """读取 js 文件，出错时返回一个小的 console.warn 脚本，避免页面崩溃"""
            if js_dir is None:
                logger.warning(f"无法找到本地 js 目录；候选: {_candidate_dirs}")
                return f'console.warn("内联 JS {fname} 失败：未找到静态 js 目录");'
            p = js_dir / fname
            try:
                text = p.read_text(encoding="utf-8")
                # 如果文件很大，可在此做额外处理（例如去掉 sourceMapping 注释），但通常不需要
                return text
            except Exception as e:
                logger.exception(f"读取本地 JS 失败: {p} -> {e}")
                return f'console.warn("内联 JS {fname} 读取失败: {str(e)}");'

        # 读取三份常用文件（如果你只需要 echarts，可以只读取 echarts.min.js）
        _echarts_js = _read_js_or_warn("echarts.min.js")
        _chartjs_js = _read_js_or_warn("chart.umd.js")
        _color_js = _read_js_or_warn("color.min.js")

        # 替换模板中可能的 <script src=...> 引用为内联脚本（支持 /static 和 ./static 两种写法）
        def _inline_js_in_template(template_str: str, filename: str, content: str) -> str:
            patterns = [
                f'<script src="/static/js/{filename}"></script>',
                f"<script src=\"/static/js/{filename}\"></script>",
                f"<script src='./static/js/{filename}'></script>",
                f'<script src="./static/js/{filename}"></script>',
                f"<script src='static/js/{filename}'></script>",
                f'<script src="static/js/{filename}"></script>',
            ]
            script_tag = f"<script>\n{content}\n</script>"
            for p in patterns:
                if p in template_str:
                    template_str = template_str.replace(p, script_tag)
            return template_str

        html_template = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    html,body{margin:0;padding:0;height:100%;overflow:hidden;}
    .chart-wrapper{position:relative;width:100vw;left:50%;transform:translateX(-50%);height:__HEIGHT__px;}
    #__CHART_ID__{width:100%;height:100%;}
    .__PANEL_ID__{
      position:absolute;top:14px;right:24px;background:rgba(255,255,255,0.98);
      padding:8px 10px;border:1px solid #ddd;border-radius:6px;box-shadow:0 6px 20px rgba(0,0,0,.12);
      z-index:9999;font-size:13px;width:180px;max-height:280px;overflow:hidden;
      backdrop-filter: blur(4px);user-select:none;cursor:move;
      display:flex;flex-direction:column;
    }
    .__PANEL_ID__.collapsed .panel-body{display:none;}
    .__PANEL_ID__.collapsed .panel-header .toggle-btn::after{content:'展开';}
    .__PANEL_ID__ .panel-header{
      display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;
      font-weight:600;cursor:pointer;
    }
    .__PANEL_ID__ .toggle-btn{font-size:12px;color:#666;}
    .__PANEL_ID__ .panel-body{flex:1;overflow-y:auto;margin-right:-10px;padding-right:10px;}
    .__PANEL_ID__ .item{display:flex;align-items:center;margin:4px 0;}
    .__PANEL_ID__ .item input{margin-right:6px;}
    .__PANEL_ID__ .color{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;}
    .__PANEL_ID__ .buttons{margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;}
    .__PANEL_ID__ button{padding:4px 6px;font-size:11px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#fafafa;}
    .__PANEL_ID__ button:hover{background:#f0f0f0;}
  </style>
</head>
<body>
  <div class="chart-wrapper">
    <div id="__CHART_ID__"></div>
    <div class="__PANEL_ID__" id="__PANEL_ID__">
      <div class="panel-header">
        <div class="title">筛选</div>
        <div class="toggle-btn">折叠</div>
      </div>
      <div class="panel-body" id="legend-items"></div>
      <div class="buttons">
        <button id="select-all">全选</button>
        <button id="deselect-all">取消</button>
        <button id="toggle-labels">隐藏数值</button>
      </div>
    </div>
  </div>

  <script src="/static/js/echarts.min.js"></script>
  <script>
  (function(){
    const raw = __RAW_JSON__;
    const uniqLabels = __UNIQLABELS_JSON__;
    const sortedMerged = __SORTEDMERGED_JSON__;
    const weeks = __WEEKS_JSON__;
    const groupsPerLevel = __GROUPSPERLEVEL_JSON__;
    const grid = {left: __GRID_LEFT__, right: __GRID_RIGHT__, top: __GRID_TOP__, bottom: __GRID_BOTTOM__};
    const palette = __PALETTE_JSON__;
    const chart = echarts.init(document.getElementById('__CHART_ID__'));

    // 控制数值显示
    let showLabels = true;

    // 生成 series
    function makeSeries(showLabelsFlag){
      return weeks.map((wk, i) => ({
        name: wk,
        type: 'bar',
        data: sortedMerged[i],
        itemStyle: { color: palette[i % palette.length] },
        barMaxWidth: 28,
        label: { show: showLabelsFlag, position: 'top', fontSize: 11, color: '#333', formatter: p => p.value > 0 ? p.value : '' }
      }));
    }

    // 首次 setOption（不包含 graphic）
    chart.setOption({
      title: { text: raw.title, left: 'center' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { show: false },
      grid: __GRID__,
      yAxis: { type: 'value', name: raw.yLabel },
      xAxis: { type: 'category', data: uniqLabels.map((_,i)=>String(i)), axisLabel: { show: false } },
      series: makeSeries(showLabels)
    });

    // splitTextIntoLines: 按空格优先换行，再按字符切分，限制行数并加省略号
    function splitTextIntoLines(text, maxWidth, font, maxLines){
      if(!text) return [];
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      ctx.font = font;
      if(ctx.measureText(text).width <= maxWidth) return [text];
      const words = text.split(' ');
      const lines = [];
      let current = words[0] || '';
      for(let i=1;i<words.length;i++){
        const test = current + ' ' + words[i];
        if(ctx.measureText(test).width <= maxWidth) current = test;
        else { lines.push(current); current = words[i]; }
      }
      lines.push(current);
      for(let i=0;i<lines.length;i++){
        if(ctx.measureText(lines[i]).width > maxWidth){
          const s = lines[i];
          lines.splice(i, 1);
          let cur = '';
          for(const ch of s){
            if(ctx.measureText(cur + ch).width <= maxWidth) cur += ch;
            else { if(cur) lines.push(cur); cur = ch; }
          }
          if(cur) lines.push(cur);
        }
      }
      if(lines.length > maxLines){
        const truncated = lines.slice(0, maxLines);
        const ell = '…';
        let last = truncated[truncated.length - 1];
        while(ctx.measureText(last + ell).width > maxWidth && last.length > 0) last = last.slice(0, -1);
        truncated[truncated.length - 1] = last + ell;
        return truncated;
      }
      return lines;
    }

    // draw: 计算同级最大高度并用 graphic 绘制分级横坐标
    function draw(){
      try{
        const w = chart.getWidth(), h = chart.getHeight();
        const {left, right, top, bottom} = grid;
        const innerW = w - left - right;
        const catW = innerW / Math.max(1, uniqLabels.length);
        const gfx = [];
        const bgColors = ['#f5f5f5','#f9f9f9','#fcfcfc','#fefefe'];
        const reservedTop = h - bottom;
        const fontSpec = `${__LABELFONTSIZE__}px sans-serif`;
        const maxLines = __MAX_LABEL_LINES__;
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.font = fontSpec;

        const actualMaxDepth = Math.max(1, __MAXDEPTH__);
        const levelHeights = new Array(actualMaxDepth).fill(0);
        const levelLabelsLines = new Array(actualMaxDepth).fill(null).map(()=>[]);

        for(let lvl = 0; lvl < actualMaxDepth; lvl++){
          const gs = (groupsPerLevel[lvl] || []).filter(g => g && g.label);
          if (gs.length === 0 && lvl > 0) continue;

          for(const g of gs){
            const wRect = (g.end - g.start + 1) * catW;
            const lines = splitTextIntoLines(g.label, wRect * 0.9, fontSpec, maxLines);
            const lineHeight = __LABELFONTSIZE__ * 1.2;
            const totalTextHeight = lines.length * lineHeight;
            levelLabelsLines[lvl].push({g, lines, totalTextHeight});
            levelHeights[lvl] = Math.max(levelHeights[lvl], totalTextHeight + 10);
          }
          if(levelHeights[lvl] < __GAPBETWEENAXES__) levelHeights[lvl] = __GAPBETWEENAXES__;
        }

        for(let lvl = 0; lvl < actualMaxDepth; lvl++){
          const gs = levelLabelsLines[lvl] || [];
          if (gs.length === 0) continue;

          const levelH = levelHeights[lvl];
          const bg = bgColors[lvl % bgColors.length];
          for(const item of gs){
            const g = item.g;
            const lines = item.lines;
            const totalTextHeight = item.totalTextHeight;
            const x = left + g.start * catW;
            const wRect = (g.end - g.start + 1) * catW;
            const y0 = reservedTop + ((actualMaxDepth - 1 - lvl) * levelH) + 6;
            gfx.push({type:'rect', shape:{x, y:y0, width:wRect, height:levelH}, style:{fill:bg, stroke:'#e6e6e6', lineWidth:1}, z:1});
            const lineHeight = __LABELFONTSIZE__ * 1.2;
            lines.forEach((line, i) => {
              const textY = y0 + (levelH - totalTextHeight) / 2 + lineHeight * (i + 0.8);
              gfx.push({type:'text', style:{text: line, x: x + wRect/2, y: textY, textAlign:'center', textVerticalAlign:'middle', fill:'#222', font: fontSpec}, z:2});
            });
          }
        }

        chart.setOption({ graphic: gfx });
      }catch(err){
        console.error('draw error', err);
      }
    }

    // 面板与交互逻辑
    const panel = document.getElementById('__PANEL_ID__');
    const header = panel.querySelector('.panel-header');
    const toggleBtn = panel.querySelector('.toggle-btn');
    const items = document.getElementById('legend-items');
    const selectAllBtn = document.getElementById('select-all');
    const deselectAllBtn = document.getElementById('deselect-all');
    const toggleLabelBtn = document.getElementById('toggle-labels');
    const selectedWeeks = new Set(weeks);

    // 拖拽支持
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    header.addEventListener('mousedown', e => {
      if (e.target === toggleBtn) return;
      isDragging = true;
      const rect = panel.getBoundingClientRect();
      dragOffset.x = e.clientX - rect.left;
      dragOffset.y = e.clientY - rect.top;
      panel.style.cursor = 'grabbing';
    });
    document.addEventListener('mousemove', e => {
      if (!isDragging) return;
      panel.style.left = `${e.clientX - dragOffset.x}px`;
      panel.style.top = `${e.clientY - dragOffset.y}px`;
      panel.style.right = 'auto';
      panel.style.transform = 'none';
    });
    document.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        panel.style.cursor = 'move';
      }
    });

    // 折叠/展开
    header.addEventListener('click', e => {
      if (e.target !== toggleBtn) return;
      panel.classList.toggle('collapsed');
    });

    function renderLegend(){
      items.innerHTML = '';
      weeks.forEach((wk, i) => {
        const checked = selectedWeeks.has(wk);
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
          <input type="checkbox" data-index="${i}" ${checked ? 'checked' : ''}>
          <span class="color" style="background:${palette[i % palette.length]}"></span>
          <span>${wk}</span>
        `;
        items.appendChild(div);
      });
    }

    function updateSeries(){
      const newSeries = weeks.map((wk, i) => ({
        name: wk,
        type: 'bar',
        data: selectedWeeks.has(wk) ? sortedMerged[i] : sortedMerged[i].map(() => 0),
        itemStyle: { color: palette[i % palette.length] },
        barMaxWidth: 28,
        label: { show: showLabels, position: 'top', fontSize: 11, color: '#333', formatter: p => p.value > 0 ? p.value : '' }
      }));
      chart.setOption({ series: newSeries });
      setTimeout(() => { draw(); }, 10);
    }

    items.addEventListener('change', e => {
      if(!e.target.matches('input[type="checkbox"]')) return;
      const idx = +e.target.dataset.index;
      const wk = weeks[idx];
      if(e.target.checked) selectedWeeks.add(wk); else selectedWeeks.delete(wk);
      updateSeries();
    });

    selectAllBtn.onclick = () => { weeks.forEach(wk => selectedWeeks.add(wk)); renderLegend(); updateSeries(); };
    deselectAllBtn.onclick = () => { selectedWeeks.clear(); renderLegend(); updateSeries(); };
    toggleLabelBtn.onclick = () => {
      showLabels = !showLabels;
      toggleLabelBtn.textContent = showLabels ? '隐藏数值' : '显示数值';
      updateSeries();
    };

    // 初始渲染
    renderLegend();
    updateSeries();

    window.addEventListener('resize', () => { chart.resize(); setTimeout(draw, 120); });

  })();
  </script>
</body>
</html>
"""
        # 在定义 html_template 之后，做内联替换
        html_template = _inline_js_in_template(html_template, "echarts.min.js", _echarts_js)
        html_template = _inline_js_in_template(html_template, "chart.umd.js", _chartjs_js)
        html_template = _inline_js_in_template(html_template, "color.min.js", _color_js)

        # 替换占位符
        replacements = {
            "__TITLE__": json.dumps(raw.get("title", "多层级柱状图"), ensure_ascii=False).strip('"'),
            "__HEIGHT__": str(total_height),
            "__CHART_ID__": chart_id,
            "__PANEL_ID__": panel_id,
            "__RAW_JSON__": json.dumps(raw, ensure_ascii=False),
            "__UNIQLABELS_JSON__": json.dumps(uniqLabels, ensure_ascii=False),
            "__SORTEDMERGED_JSON__": json.dumps(sortedMerged, ensure_ascii=False),
            "__WEEKS_JSON__": json.dumps(weeks, ensure_ascii=False),
            "__GROUPSPERLEVEL_JSON__": json.dumps(groupsPerLevel, ensure_ascii=False),
            "__PALETTE_JSON__": json.dumps(palette, ensure_ascii=False),
            "__GRID_LEFT__": str(grid["left"]),
            "__GRID_RIGHT__": str(grid["right"]),
            "__GRID_TOP__": str(grid["top"]),
            "__GRID_BOTTOM__": str(grid["bottom"]),
            "__LABELFONTSIZE__": str(labelFontSize),
            "__GAPBETWEENAXES__": str(gapBetweenAxes),
            "__MAX_LABEL_LINES__": str(maxLabelLines),
            "__MAXDEPTH__": str(maxDepth),
            "__GRID__": json.dumps(grid)
        }

        html = html_template
        for k, v in replacements.items():
            html = html.replace(k, v)

        return html, total_height

    except Exception as e:
        logger.error(f"hierarchical_chart 渲染失败: {e}\n输入: {json.dumps(viz_data, ensure_ascii=False)[:1500]}")
        return f"<div style='color:red;padding:10px;'>图表渲染失败: {str(e)}</div>", height


__all__ = ["render_hierarchical_bar"]


