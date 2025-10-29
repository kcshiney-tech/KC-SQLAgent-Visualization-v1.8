# hierarchical_chart.py
"""
终极健壮版多层级柱状图渲染器
- 自动补全 title / yLabel / xLabel
- 支持 labels + values[] 格式
- 支持 values.label 重复 → 自动合并
- 支持 Chart.js config 中的 raw_data
- 可勾选图例 + 全选/取消
- 高度自适应（900px）
"""

import json
import uuid
from typing import Dict, Any, List, Tuple
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


def _detect_week_in_label(labels: List[str]) -> Tuple[bool, List[str], List[str]]:
    """检测 .2025-XX 或 .第XX周"""
    if not labels:
        return False, [], []

    # 支持两种格式
    pattern1 = re.compile(r'\.(\d{4}-\d{1,2})$')  # .2025-41
    pattern2 = re.compile(r'\.(第\d{1,2}周)$')   # .第41周

    weeks = []
    clean_labels = []

    for lbl in labels:
        m1 = pattern1.search(lbl)
        m2 = pattern2.search(lbl)
        m = m1 or m2
        if m:
            week = m.group(1)
            clean_lbl = lbl[:m.start()]
            weeks.append(week)
            clean_labels.append(clean_lbl)
        else:
            weeks.append(None)
            clean_labels.append(lbl)

    all_have_week = all(w is not None for w in weeks)
    return all_have_week, clean_labels, weeks if all_have_week else []


def render_hierarchical_bar(viz_data: Dict[str, Any], height: int = 600) -> str:
    """
    渲染多层级柱状图（ECharts） - 终极健壮版
    """
    chart_id = f"echarts_{uuid.uuid4().hex}"
    panel_id = f"legend_panel_{uuid.uuid4().hex[:8]}"

    try:
        # ==================== 1. 数据标准化 ====================
        raw = _normalize_data(viz_data)
        logger.debug(f"hierarchical_chart 接收数据: title={raw['title']}, labels={len(raw['labels'])}, values={len(raw['values'])}")

        # ==================== 2. 自动检测“周在标签末尾” ====================
        is_week_in_label, clean_labels, week_labels = _detect_week_in_label(raw["labels"])
        values_per_series = raw["values"]

        if is_week_in_label:
            unique_weeks = sorted(set(week_labels), key=lambda x: int(re.search(r'\d+', x).group()))
            week_to_idx = {w: i for i, w in enumerate(unique_weeks)}
            clean_to_final_idx = {lbl: i for i, lbl in enumerate(clean_labels)}

            series_data = [[] for _ in unique_weeks]
            orig_label_to_idx = {lbl: i for i, lbl in enumerate(raw["labels"])}

            for clean_lbl, week in zip(clean_labels, week_labels):
                final_idx = clean_to_final_idx[clean_lbl]
                series_idx = week_to_idx[week]
                while len(series_data[series_idx]) <= final_idx:
                    series_data[series_idx].append(0)
                orig_lbl = f"{clean_lbl}.{week}"
                orig_idx = orig_label_to_idx.get(orig_lbl)
                if orig_idx is not None:
                    value = values_per_series[0]["data"][orig_idx]
                    series_data[series_idx][final_idx] = value

            max_len = len(clean_to_final_idx)
            for s in series_data:
                s.extend([0] * (max_len - len(s)))

            raw["labels"] = list(clean_to_final_idx.keys())
            raw["values"] = [{"label": week, "data": series_data[i]} for i, week in enumerate(unique_weeks)]

        # ==================== 3. 去重 + 合并（取非零） ====================
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

        deduped = dedupe_merge_first_non_zero(raw["labels"], raw["values"])
        uniqLabels = deduped["uniqueLabels"]
        merged = deduped["merged"]
        weeks = deduped["weeks"]

        # ==================== 4. 层级拆分 + 排序 ====================
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

        sorted_info = hierarchical_sort(uniqLabels)
        order = sorted_info["indices"]
        sortedRows = sorted_info["rows"]
        maxDepth = sorted_info["maxDepth"]

        uniqLabels = [uniqLabels[i] for i in order]
        sortedMerged = [[row[i] for i in order] for row in merged]

        # ==================== 5. 递归分组 + 分割线 ====================
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
            return rec(0, 0, len(rows) - 1) if rows else []

        root = build_hierarchy(sortedRows)

        def collect_groups(root, depth):
            out = [[] for _ in range(depth)]
            def dfs(nodes, l):
                for n in nodes:
                    out[l].append({"start": n["start"], "end": n["end"], "label": n["label"]})
                    if n["children"]:
                        dfs(n["children"], l + 1)
            dfs(root, 0)
            return out

        groupsPerLevel = collect_groups(root, maxDepth)

        def split_lines():
            if not uniqLabels:
                return []
            arr = [None] * (len(uniqLabels) + 1)
            arr[0] = arr[-1] = maxDepth
            def mark(p, l):
                if 0 <= p < len(arr) and (arr[p] is None or l < arr[p]):
                    arr[p] = l
            def walk(nodes, l):
                for n in nodes:
                    mark(n["start"], l)
                    mark(n["end"] + 1, l)
                    if n["children"]:
                        walk(n["children"], l + 1)
                    else:
                        for p in range(n["start"], n["end"]):
                            mark(p + 1, maxDepth)
            walk(root, 0)
            return arr

        splitAt = split_lines()

        # ==================== 6. ECharts 配置 ====================
        labelFontSize = 11
        gapBetweenAxes = 30
        extraBottomPadding = 10
        gridBase = {"left": 5, "right": 5, "top": 80, "bottom": 50}  # 右边留空间给图例面板
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

        # ==================== 7. HTML 模板（可勾选图例） ====================
        html_template = f"""
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width,initial-scale=1">
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
            <div class="title">筛选周</div>
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