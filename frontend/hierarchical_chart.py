# hierarchical_chart.py
"""
终极健壮版多层级柱状图渲染器
- 自动补全 title / yLabel / xLabel
- 支持 labels + values[] 格式
- 支持 values.label 重复 → 自动合并
- 支持 Chart.js config 中的 raw_data
- 可勾选图例 + 全选/取消
- 高度自适应（900px）
- 通用最里层标签提取：任意 .xxx 作为 series，字符升序排序，右上角筛选框
"""

import json
import uuid
from typing import Dict, Any, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


def _safe_get(data: Dict, key: str, default: Any = None):
    return data.get(key, default)


def _normalize_data(viz_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化数据 + 自动补全 + 兼容 Chart.js config
    """
    # 1. 兼容 Chart.js config
    if "data" in viz_data and "datasets" in viz_data.get("data", {}):
        cfg = viz_data
        raw_data = cfg.get("raw_data")
        if raw_data:
            viz_data = raw_data
        else:
            viz_data = {
                "title": cfg["options"]["plugins"]["title"]["text"],
                "xLabel": cfg["options"]["scales"]["x"]["title"]["text"],
                "yLabel": cfg["options"]["scales"]["y"]["title"]["text"],
                "labels": cfg["data"]["labels"],
                "values": [
                    {"label": ds["label"], "data": ds["data"]}
                    for ds in cfg["data"]["datasets"]
                ]
            }

    raw = json.loads(json.dumps(viz_data))

    # 2. 自动补全
    raw["title"] = raw.get("title") or "多层级柱状图"
    raw["yLabel"] = raw.get("yLabel") or "数值"
    raw["xLabel"] = raw.get("xLabel") or "分类"

    # 3. 验证 labels + values
    if "labels" not in raw or "values" not in raw:
        raise ValueError("viz_data 必须包含 labels 和 values")

    # 4. 合并重复 label 的 values
    merged_values = {}
    for v in raw["values"]:
        label = v["label"]
        data = v["data"]
        if label in merged_values:
            # 合并：取非零值优先
            old = merged_values[label]
            merged = [a if a != 0 else b for a, b in zip(old, data)]
            merged = [b if a == 0 and b != 0 else a for a, b in zip(merged, data)]
            merged_values[label] = merged
        else:
            merged_values[label] = data

    raw["values"] = [
        {"label": label, "data": data}
        for label, data in merged_values.items()
    ]

    return raw


def _detect_week_in_label(labels: List[str]) -> Tuple[bool, List[str], List[Optional[int]]]:
    """
    检测标签末尾是否包含“周数字”，返回：
    - all_have_week: bool
    - clean_labels: List[str]   # 去掉 .周 的部分
    - week_numbers: List[Optional[int]]  # 提取的数字（无则 None）
    """
    patterns = [
        r'\.(\d{4}-\d{1,2})$',           # .2025-41
        r'\.(第\d{1,2}周)$',             # .第41周
        r'\.(w|week|W|Week)?(\d{1,2})$', # .W41 / .week41 / .41
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    clean_labels = []
    week_numbers = []

    for lbl in labels:
        matched = False
        for pat in compiled:
            m = pat.search(lbl)
            if m:
                # 提取最后一个组的字符串
                num_str = m.group(m.lastindex) if m.lastindex else m.group(0)
                # 提取所有数字，取最后一个（处理 '2025-41' → '41'）
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


def _extract_innermost_as_series(
    labels: List[str], 
    values: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    提取所有标签的 **最里层**（最后一个 . 之后的部分）作为 series
    - 自动去重
    - 按字符串升序排序
    - 合并相同 clean_label 的值（取非0）
    """
    if not labels or '.' not in labels[0]:
        return {
            "is_innermost_series": False,
            "clean_labels": labels,
            "series_list": values
        }

    # 1. 拆分最里层
    clean_to_orig = {}      # clean_label -> [orig_idx]
    innermost_parts = []    # 最里层原始文本
    clean_labels = []

    for i, lbl in enumerate(labels):
        if '.' not in lbl:
            clean = lbl
            inner = ""
        else:
            clean, inner = lbl.rsplit('.', 1)
        clean = clean.strip()
        inner = inner.strip()

        if clean not in clean_to_orig:
            clean_to_orig[clean] = []
            clean_labels.append(clean)
        clean_to_orig[clean].append(i)
        innermost_parts.append(inner)

    # 2. 提取唯一最里层 + 按字符串升序
    unique_innermost = sorted(set(innermost_parts), key=lambda x: x.lower())

    # 3. 构建 series_data
    n_clean = len(clean_labels)
    n_series = len(unique_innermost)
    series_data = [[0] * n_clean for _ in range(n_series)]

    inner_to_idx = {inner: idx for idx, inner in enumerate(unique_innermost)}

    # 4. 填充数据（支持多 values）
    for v in values:
        data = v["data"]
        for clean, orig_indices in clean_to_orig.items():
            c_idx = clean_labels.index(clean)
            for orig_idx in orig_indices:
                if orig_idx >= len(data) or data[orig_idx] == 0:
                    continue
                inner = innermost_parts[orig_idx]
                if inner not in inner_to_idx:
                    continue
                s_idx = inner_to_idx[inner]
                if series_data[s_idx][c_idx] == 0:  # 取第一个非0
                    series_data[s_idx][c_idx] = data[orig_idx]

    series_list = [
        {"label": unique_innermost[i], "data": series_data[i]}
        for i in range(n_series)
    ]

    return {
        "is_innermost_series": True,
        "clean_labels": clean_labels,
        "series_list": series_list
    }


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


def split_lines():
    # 简化：返回空列表，实际可根据需要实现
    return []


def render_hierarchical_bar(viz_data: Dict[str, Any], height: int = 600) -> str:
    """
    渲染多层级柱状图（ECharts） - 终极健壮版
    """
    chart_id = f"echarts_{uuid.uuid4().hex}"
    panel_id = f"legend_panel_{uuid.uuid4().hex[:8]}"

    try:
        # ==================== 1. 数据标准化 ====================
        raw = _normalize_data(viz_data)

        # ==================== 2. 通用最里层提取（任意标签作为系列） ====================
        extracted = _extract_innermost_as_series(raw["labels"], raw["values"])

        if extracted["is_innermost_series"]:
            uniqLabels = extracted["clean_labels"]
            merged = [s["data"] for s in extracted["series_list"]]
            weeks = [s["label"] for s in extracted["series_list"]]  # 通用: 最里层标签
        else:
            # 回退: 周特定逻辑（如果最里层是周）
            is_week_in_label, clean_labels, week_numbers = _detect_week_in_label(raw["labels"])
            if is_week_in_label:
                # 周升序（数字排序）
                assert all(n is not None for n in week_numbers), "所有标签必须包含周号"
                unique_weeks_with_num = sorted(
                    set(zip(raw["labels"], week_numbers)),
                    key=lambda x: x[1] if x[1] is not None else float('inf')
                )
                # unique_weeks_with_num = sorted(set(zip(raw["labels"], week_numbers)), key=lambda x: x[1])
                # unique_weeks_with_num = sorted(set(zip(raw["labels"], week_numbers)), key=lambda x: x[1])
                unique_weeks = [w for w, _ in unique_weeks_with_num]
                week_to_idx = {w: i for i, w in enumerate(unique_weeks)}

                clean_to_idx = {}
                for i, cl in enumerate(clean_labels):
                    if cl not in clean_to_idx:
                        clean_to_idx[cl] = len(clean_to_idx)
                n_clean = len(clean_to_idx)

                series_data = [[0] * n_clean for _ in unique_weeks]

                for series in raw["values"]:
                    data = series["data"]
                    for j, orig_lbl in enumerate(raw["labels"]):
                        if j >= len(data) or data[j] == 0:
                            continue
                        clean_lbl = orig_lbl.rsplit(".", 1)[0]
                        if clean_lbl not in clean_to_idx:
                            continue
                        c_idx = clean_to_idx[clean_lbl]
                        w_idx = week_to_idx[orig_lbl]
                        if series_data[w_idx][c_idx] == 0:
                            series_data[w_idx][c_idx] = data[j]

                uniqLabels = list(clean_to_idx.keys())
                merged = series_data
                weeks = unique_weeks
            else:
                # 无最里层: 原去重逻辑
                deduped = dedupe_merge_first_non_zero(raw["labels"], raw["values"])
                uniqLabels = deduped["uniqueLabels"]
                merged = deduped["merged"]
                weeks = deduped["weeks"]

        # ==================== 3. 层级排序 + 分组 ====================
        sorted_info = hierarchical_sort(uniqLabels)
        order = sorted_info["indices"]
        sortedRows = sorted_info["rows"]
        maxDepth = sorted_info["maxDepth"]

        uniqLabels = [uniqLabels[i] for i in order]
        sortedMerged = [[row[i] for i in order] for row in merged]

        root = build_hierarchy(sortedRows)
        groupsPerLevel = collect_groups(root, maxDepth)
        splitAt = split_lines()

        # ==================== 4. ECharts 配置（优化纵坐标显示） ====================
        labelFontSize = 11
        gapBetweenAxes = 30
        extraBottomPadding = 10
        gridBase = {"left": 70, "right": 70, "top": 80, "bottom": 80}  # 加大 left/right/bottom 以显示完整 Y 轴/图例
        reserved = maxDepth * gapBetweenAxes + extraBottomPadding
        grid = {**gridBase, "bottom": gridBase["bottom"] + reserved}

        palette = ['#5470c6','#91cc75','#fac858','#ee6666','#73c0de','#3ba272','#fc8452','#9a60b4','#5ab1ef']
        series = [
            {
                "name": weeks[i],
                "type": "bar",
                "data": sortedMerged[i],
                "itemStyle": {"color": palette[i % len(palette)]},
                "barMaxWidth": 28
            }
            for i in range(len(weeks))
        ]

        # ==================== 5. HTML 模板 ====================
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>{raw["title"]}</title>
          <style>
            #{chart_id} {{ width:100%; height:{height}px; }}
            .{panel_id} {{
              position: absolute; top: 20px; right: 20px; background: #fff; padding: 12px;
              border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.1);
              z-index: 100; font-size: 13px; min-width: 160px; max-height: 80vh; overflow-y: auto;
            }}
            .{panel_id} .title {{ margin: 0 0 8px 0; font-weight: bold; color: #333; }}
            .{panel_id} .item {{ display: flex; align-items: center; margin: 4px 0; }}
            .{panel_id} .item input {{ margin-right: 6px; }}
            .{panel_id} .color {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; border-radius: 2px; }}
            .{panel_id} .buttons {{ margin-top: 10px; display: flex; gap: 6px; }}
            .{panel_id} .buttons button {{ flex: 1; padding: 4px 8px; font-size: 12px; cursor: pointer; border: 1px solid #ccc; border-radius: 4px; background: #f9f9f9; }}
            .{panel_id} .buttons button:hover {{ background: #f0f0f0; }}
          </style>
        </head>
        <body>
          <div id="{chart_id}"></div>
          <div class="{panel_id}" id="{panel_id}">
            <div class="title">筛选</div>
            <div id="legend-items"></div>
            <div class="buttons">
              <button id="select-all">全选</button>
              <button id="deselect-all">取消</button>
            </div>
          </div>

          <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
          <script>
          (() => {{
            const raw = {json.dumps(raw, ensure_ascii=False)};
            const uniqLabels = {json.dumps(uniqLabels)};
            const sortedMerged = {json.dumps(sortedMerged)};
            const weeks = {json.dumps(weeks)};
            const maxDepth = {maxDepth};
            const groupsPerLevel = {json.dumps(groupsPerLevel)};
            const splitAt = {json.dumps(splitAt)};
            const palette = {json.dumps(palette)};

            const reserved = maxDepth * {gapBetweenAxes} + {extraBottomPadding};
            const grid = {{left: {gridBase['left']}, right: {gridBase['right']}, top: {gridBase['top']}, bottom: {gridBase['bottom']} + reserved}};

            const chart = echarts.init(document.getElementById('{chart_id}'));
            const series = weeks.map((wk, i) => ({{
              name: wk, type: 'bar', data: sortedMerged[i],
              itemStyle: {{color: palette[i % palette.length]}}, barMaxWidth: 28
            }}));

            chart.setOption({{
              title: {{text: raw.title, left: 'center'}},
              tooltip: {{trigger: 'axis', axisPointer: {{type: 'shadow'}},
                formatter: p => {{
                  if (!p || !p.length) return '';
                  const idx = p[0].dataIndex;
                  const lbl = uniqLabels[idx];
                  let html = `<b>${{lbl}}</b><br/>`;
                  p.forEach(o => html += `<span style="display:inline-block;width:10px;height:10px;background:${{o.color}};margin-right:6px;border-radius:2px;"></span>${{o.seriesName}}: ${{o.data}}<br/>`);
                  return html;
                }}
              }},
              legend: {{show: false}},
              grid, 
              xAxis: {{type: 'category', data: uniqLabels.map((_,i)=>i+''), axisLine:{{show:false}}, axisTick:{{show:false}}, axisLabel:{{show:false}}}},
              yAxis: {{type: 'value', name: raw.yLabel}},
              series
            }});

            // 背景层级线
            function draw() {{
              const w = chart.getWidth(), h = chart.getHeight();
              const {{left, right, top, bottom}} = grid;
              const innerW = w - left - right;
              const catW = innerW / uniqLabels.length;
              const gfx = [];
              const bgColors = ['#f5f5f5', '#f9f9f9', '#fcfcfc', '#fefefe'];
              const reservedTop = h - bottom;

              for (let lvl = 0; lvl < maxDepth; lvl++) {{
                const gs = groupsPerLevel[lvl];
                const bg = bgColors[lvl % bgColors.length];
                const y0 = reservedTop + ((maxDepth - 1 - lvl) * {gapBetweenAxes}) + 6;
                const hBand = {gapBetweenAxes} - 8;
                gs.forEach(g => {{
                  if (!g.label) return;
                  const x = left + g.start * catW;
                  const w = (g.end - g.start + 1) * catW;
                  gfx.push({{type:'rect', shape:{{x, y:y0, width:w, height:hBand}}, style:{{fill:bg, stroke:'#e6e6e6', lineWidth:1}}, z:1}});
                  gfx.push({{type:'text', style:{{text:g.label, x:x+w/2, y:y0+hBand/2, textAlign:'center', textVerticalAlign:'middle', fill:'#222', font:`{labelFontSize}px sans-serif`}}, z:2}});
                }});
              }}

              const lineTop = top - 6;
              splitAt.forEach((lvl, pos) => {{
                if (lvl === null) return;
                const x = left + pos * catW;
                const lineEnd = reservedTop + (maxDepth - lvl) * {gapBetweenAxes};
                gfx.push({{type:'line', shape:{{x1:x, y1:lineTop, x2:x, y2:lineEnd}}, style:{{stroke:'#ccc', lineWidth:1}}, z:3}});
              }});

              chart.setOption({{graphic: gfx}});
            }}
            draw();

            // 可勾选图例
            const panel = document.getElementById('{panel_id}');
            const items = document.getElementById('legend-items');
            const selectAllBtn = document.getElementById('select-all');
            const deselectAllBtn = document.getElementById('deselect-all');
            const selectedWeeks = new Set(weeks);

            function renderLegend() {{
              items.innerHTML = '';
              weeks.forEach((wk, i) => {{
                const checked = selectedWeeks.has(wk);
                const div = document.createElement('div');
                div.className = 'item';
                div.innerHTML = `
                  <input type="checkbox" data-index="${{i}}" ${{checked ? 'checked' : ''}}>
                  <span class="color" style="background: ${{palette[i % palette.length]}}"></span>
                  <span>${{wk}}</span>
                `;
                items.appendChild(div);
              }});
            }}

            function updateSeries() {{
              const newSeries = weeks.map((wk, i) => {{
                const show = selectedWeeks.has(wk);
                return {{
                  ...series[i],
                  data: show ? sortedMerged[i] : sortedMerged[i].map(() => 0),
                  barWidth: show ? undefined : 0.1
                }};
              }});
              chart.setOption({{series: newSeries}});
            }}

            items.addEventListener('change', e => {{
              if (!e.target.matches('input[type="checkbox"]')) return;
              const idx = +e.target.dataset.index;
              const wk = weeks[idx];
              if (e.target.checked) selectedWeeks.add(wk);
              else selectedWeeks.delete(wk);
              updateSeries();
            }});

            selectAllBtn.onclick = () => {{ weeks.forEach(wk => selectedWeeks.add(wk)); renderLegend(); updateSeries(); }};
            deselectAllBtn.onclick = () => {{ selectedWeeks.clear(); renderLegend(); updateSeries(); }};

            renderLegend();
            updateSeries();

            window.addEventListener('resize', () => {{ chart.resize(); setTimeout(draw, 120); }});
          }})();
          </script>
        </body>
        </html>
        """
        return html_template

    except Exception as e:
        logger.error(f"hierarchical_chart 渲染失败: {e}\n输入: {json.dumps(viz_data)[:500]}")
        return f"<div style='padding:20px;color:red;'>图表渲染失败: {str(e)[:200]}</div>"


__all__ = ["render_hierarchical_bar"]