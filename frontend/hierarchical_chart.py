# hierarchical_chart.py
"""
独立的多层级柱状图渲染模块
输入：LLM 产生的 hierarchical_bar JSON（与 viz_utils 中 graph_instructions 完全一致）
输出：完整的 <iframe> HTML 字符串（Streamlit 用 components.html 直接展示）
"""

# hierarchical_chart.py
"""
独立的多层级柱状图渲染模块（终极健壮版）
支持：
1. 标准格式：labels + values[]
2. Chart.js 格式：datasets[]
3. 周在标签末尾自动提取
4. 任意层级、去重、补零、自然排序
"""

import json
import uuid
from typing import Dict, Any, List, Tuple
import re


def _normalize_data(viz_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一数据格式：
    - 支持 labels + values[]
    - 支持 datasets[]（Chart.js 风格）
    - 自动转换
    """
    raw = json.loads(json.dumps(viz_data))  # 深拷贝

    # 必须字段
    required = ["title", "yLabel"]
    for key in required:
        if key not in raw:
            raise ValueError(f"viz_data 缺少必填字段: {key}")

    # 情况1：已有 labels + values
    if "labels" in raw and "values" in raw:
        if not isinstance(raw["labels"], list) or not isinstance(raw["values"], list):
            raise ValueError("labels 和 values 必须是列表")
        return raw

    # 情况2：datasets 格式
    if "datasets" in raw:
        if not isinstance(raw["datasets"], list):
            raise ValueError("datasets 必须是列表")
        raw["values"] = [
            {"label": ds.get("label", f"Series {i}"), "data": ds.get("data", [])}
            for i, ds in enumerate(raw["datasets"])
        ]
        raw["labels"] = raw.get("labels", [f"Item {i}" for i in range(len(raw["values"][0]["data"]))])
        return raw

    raise ValueError("viz_data 必须包含 labels+values 或 datasets")


def _detect_week_in_label(labels: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    检测 labels 是否以 .2025-XX 结尾
    返回: (is_week_in_label, clean_labels, week_labels)
    """
    if not labels:
        return False, [], []

    week_pattern = re.compile(r'\.(\d{4}-\d{1,2})$')
    weeks = []
    clean_labels = []

    for lbl in labels:
        m = week_pattern.search(lbl)
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


def render_hierarchical_bar(viz_data: Dict[str, Any], height: int = 500) -> str:
    """
    渲染多层级柱状图（ECharts）
    """
    chart_id = f"echarts_{uuid.uuid4().hex}"

    try:
        # ==================== 1. 数据标准化 ====================
        raw = _normalize_data(viz_data)

        # ==================== 2. 自动检测“周在标签末尾” ====================
        is_week_in_label, clean_labels, week_labels = _detect_week_in_label(raw["labels"])
        values_per_series = raw["values"]

        if is_week_in_label:
            # 新格式：周在标签中 → 构建 series
            unique_weeks = sorted(set(week_labels), key=lambda x: int(x.split('-')[1]))
            week_to_idx = {w: i for i, w in enumerate(unique_weeks)}

            # 构建 label → 原始索引映射
            label_to_orig_idx = {lbl: i for i, lbl in enumerate(raw["labels"])}

            # 初始化 series data
            series_data = [[] for _ in unique_weeks]
            clean_to_final_idx = {}

            # 填充数据
            for clean_lbl, week in zip(clean_labels, week_labels):
                if clean_lbl not in clean_to_final_idx:
                    clean_to_final_idx[clean_lbl] = len(clean_to_final_idx)
                final_idx = clean_to_final_idx[clean_lbl]
                series_idx = week_to_idx[week]

                # 扩展 series_data
                while len(series_data[series_idx]) <= final_idx:
                    series_data[series_idx].append(0)

                orig_idx = label_to_orig_idx[f"{clean_lbl}.{week}"]
                value = values_per_series[0]["data"][orig_idx]
                series_data[series_idx][final_idx] = value

            # 补 0
            max_len = len(clean_to_final_idx)
            for s in series_data:
                s.extend([0] * (max_len - len(s)))

            # 更新 raw
            raw["labels"] = list(clean_to_final_idx.keys())
            raw["values"] = [
                {"label": week, "data": series_data[i]}
                for i, week in enumerate(unique_weeks)
            ]
        else:
            # 旧格式：values[i].label 是 series
            pass

        # ==================== 3. 去重 + 合并（取第一个非零） ====================
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

            # 补全 0
            for w in range(week_cnt):
                while len(merged[w]) < len(uniq):
                    merged[w].append(0)

            weeks = [v["label"] for v in values_per_week]
            return {"uniqueLabels": uniq, "merged": merged, "weeks": weeks}

        deduped = dedupe_merge_first_non_zero(raw["labels"], raw["values"])
        uniqLabels = deduped["uniqueLabels"]
        merged = deduped["merged"]
        weeks = deduped["weeks"]

        # ==================== 4. 层级拆分 + 自然排序 ====================
        def split_rows(labels):
            rows = [s.split('.') for s in labels]
            max_d = max(len(r) for r in rows) if rows else 0
            for r in rows:
                r.extend([''] * (max_d - len(r)))
            return {"rows": rows, "maxDepth": max_d}

        def hierarchical_sort(labels):
            split = split_rows(labels)
            rows, maxDepth = split["rows"], split["maxDepth"]
            idx = list(range(len(labels)))
            idx.sort(key=lambda i: tuple(rows[i]))
            return {
                "indices": idx,
                "rows": [rows[i] for i in idx],
                "maxDepth": maxDepth
            }

        sorted_info = hierarchical_sort(uniqLabels)
        order = sorted_info["indices"]
        sortedRows = sorted_info["rows"]
        maxDepth = sorted_info["maxDepth"]

        uniqLabels = [uniqLabels[i] for i in order]
        sortedMerged = [[row[i] for i in order] for row in merged]

        # ==================== 5. 递归分组 ====================
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

        # ==================== 6. 分割线位置 ====================
        def split_lines():
            arr = [None] * (len(uniqLabels) + 1)
            arr[0] = arr[-1] = maxDepth

            def mark(p, l):
                if arr[p] is None or l < arr[p]:
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

        # ==================== 7. ECharts 配置 ====================
        labelFontSize = 11
        gapBetweenAxes = 30
        extraBottomPadding = 10
        gridBase = {"left": 130, "right": 80, "top": 80, "bottom": 28}
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

        # ==================== 8. HTML 模板 ====================
        html_template = f"""
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width,initial-scale=1">
          <style>
            #{chart_id} {{ width:100%; height:{height}px; }}
          </style>
        </head>
        <body>
          <div id="{chart_id}"></div>

          <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
          <script>
          (() => {{
            const raw = {json.dumps(raw, ensure_ascii=False)};
            const labelFontSize = {labelFontSize};
            const gapBetweenAxes = {gapBetweenAxes};
            const extraBottomPadding = {extraBottomPadding};
            const gridBase = {json.dumps(gridBase)};
            const palette = {json.dumps(palette)};

            const uniqLabels = {json.dumps(uniqLabels)};
            const sortedMerged = {json.dumps(sortedMerged)};
            const weeks = {json.dumps(weeks)};
            const maxDepth = {maxDepth};
            const groupsPerLevel = {json.dumps(groupsPerLevel)};
            const splitAt = {json.dumps(splitAt)};

            const reserved = maxDepth * gapBetweenAxes + extraBottomPadding;
            const grid = {{...gridBase, bottom: gridBase.bottom + reserved}};

            const chart = echarts.init(document.getElementById('{chart_id}'));
            const series = weeks.map((wk, i) => ({{
              name: wk, type: 'bar', data: sortedMerged[i],
              itemStyle: {{color: palette[i % palette.length]}}, barMaxWidth: 28
            }}));

            chart.setOption({{
              title: {{text: raw.title, left: 'center'}},
              tooltip: {{
                trigger: 'axis', axisPointer: {{type: 'shadow'}},
                formatter: p => {{
                  if (!p || !p.length) return '';
                  const idx = p[0].dataIndex;
                  const lbl = uniqLabels[idx];
                  let html = `<b>${{lbl}}</b><br/>`;
                  p.forEach(o => html += `<span style="display:inline-block;width:10px;height:10px;background:${{o.color}};margin-right:6px;border-radius:2px;"></span>${{o.seriesName}}: ${{o.data}}<br/>`);
                  return html;
                }}
              }},
              legend: {{top: 36, right: grid.right - 10}},
              grid, 
              xAxis: {{type: 'category', data: uniqLabels.map((_,i)=>i+''), axisLine:{{show:false}}, axisTick:{{show:false}}, axisLabel:{{show:false}}}},
              yAxis: {{type: 'value', name: raw.yLabel}},
              series
            }});

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
                const y0 = reservedTop + ((maxDepth - 1 - lvl) * gapBetweenAxes) + 6;
                const hBand = gapBetweenAxes - 8;
                gs.forEach(g => {{
                  if (!g.label) return;
                  const x = left + g.start * catW;
                  const w = (g.end - g.start + 1) * catW;
                  gfx.push({{type:'rect', shape:{{x, y:y0, width:w, height:hBand}}, style:{{fill:bg, stroke:'#e6e6e6', lineWidth:1}}, z:1}});
                  gfx.push({{type:'text', style:{{text:g.label, x:x+w/2, y:y0+hBand/2, textAlign:'center', textVerticalAlign:'middle', fill:'#222', font:`${{labelFontSize}}px sans-serif`}}, z:2}});
                }});
              }}

              const lineTop = top - 6;
              splitAt.forEach((lvl, pos) => {{
                if (lvl === null) return;
                const x = left + pos * catW;
                const lineEnd = reservedTop + (maxDepth - lvl) * gapBetweenAxes;
                gfx.push({{type:'line', shape:{{x1:x, y1:lineTop, x2:x, y2:lineEnd}}, style:{{stroke:'#ccc', lineWidth:1}}, z:3}});
              }});

              chart.setOption({{graphic: gfx}});
            }}
            draw();
            window.addEventListener('resize', () => {{ chart.resize(); setTimeout(draw, 120); }});
          }})();
          </script>
        </body>
        </html>
        """
        return html_template

    except Exception as e:
        # 详细错误日志
        import traceback
        print("hierarchical_chart.py 渲染失败:", traceback.format_exc())
        raise


# ==================== 导出函数（兼容旧调用） ====================
__all__ = ["render_hierarchical_bar"]


# def render_hierarchical_bar2(viz_data: Dict[str, Any], height: int = 500) -> str:
#     """
#     渲染多层级柱状图（ECharts 实现）

#     参数
#     ----
#     viz_data: dict
#         必须包含键：title, xLabel, yLabel, labels, values
#         结构与 viz_utils.graph_instructions["hierarchical_bar"] 完全一致
#     height: int
#         渲染高度（px）

#     返回
#     ----
#     str
#         可直接喂给 `st.components.v1.html` 的完整 HTML
#     """
#     chart_id = f"echarts_{uuid.uuid4().hex}"
#     raw_json = json.dumps(viz_data, ensure_ascii=False)

#     html_template = f"""
#     <!doctype html>
#     <html lang="zh-CN">
#     <head>
#       <meta charset="utf-8">
#       <meta name="viewport" content="width=device-width,initial-scale=1">
#       <style>
#         #{chart_id} {{ width:100%; height:{height}px; }}
#       </style>
#     </head>
#     <body>
#       <div id="{chart_id}"></div>

#       <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
#       <script>
#       (() => {{
#         // ---------- 1. 原始数据 ----------
#         const raw = {raw_json};

#         // ---------- 2. 配置 ----------
#         const labelFontSize = 11;
#         const gapBetweenAxes = 30;
#         const extraBottomPadding = 10;
#         const gridBase = {{ left:130, right:80, top:80, bottom:28 }};

#         // ---------- 3. 工具函数 ----------
#         const naturalCompare = (a,b) => a.localeCompare(b, undefined, {{numeric:true, sensitivity:'base'}});

#         // ---------- 4. 去重 + 合并（取第一个非零） ----------
#         function dedupeMergeFirstNonZero(labels, valuesPerWeek) {{
#           const map = new Map();
#           const uniq = [];
#           const weekCnt = valuesPerWeek.length;
#           const merged = Array.from({{length:weekCnt}},()=>[]);

#           for (let i=0;i<labels.length;i++) {{
#             const lbl = labels[i];
#             let idx = map.get(lbl);
#             if (idx===undefined) {{
#               idx = uniq.length;
#               uniq.push(lbl); map.set(lbl,idx);
#               for (let w=0;w<weekCnt;w++) merged[w][idx]=0;
#             }}
#             for (let w=0;w<weekCnt;w++) {{
#               const v = Number(valuesPerWeek[w].data[i]||0);
#               if (merged[w][idx]===0 && v!==0) merged[w][idx]=v;
#             }}
#           }}
#           // 补全 0
#           for (let w=0;w<weekCnt;w++) {{
#             for (let k=0;k<uniq.length;k++) if (merged[w][k]===undefined) merged[w][k]=0;
#           }}
#           return {{uniqueLabels:uniq, merged, weeks:valuesPerWeek.map(o=>o.label)}};
#         }}

#         const {{uniqueLabels:rawUniq, merged, weeks}} = dedupeMergeFirstNonZero(raw.labels, raw.values);

#         // ---------- 5. 层级拆分 + 自然排序 ----------
#         function splitRows(labels) {{
#           const rows = labels.map(s=>s.split('.'));
#           const maxD = Math.max(...rows.map(r=>r.length));
#           rows.forEach(r=>{{while(r.length<maxD) r.push('')}});
#           return {{rows, maxDepth:maxD}};
#         }}
#         function hierarchicalSort(labels) {{
#           const {{rows, maxDepth}} = splitRows(labels);
#           const idx = labels.map((_,i)=>i);
#           idx.sort((a,b)=>{{for(let l=0;l<maxDepth;l++){{const cmp=naturalCompare(rows[a][l]||'',rows[b][l]||''); if(cmp) return cmp;}} return 0;}});
#           return {{indices:idx, rows, maxDepth}};
#         }}

#         const {{indices:order, rows:sortedRows, maxDepth}} = hierarchicalSort(rawUniq);
#         const uniqLabels = order.map(i=>rawUniq[i]);
#         const sortedMerged = merged.map(w=>order.map(i=>w[i]));

#         // ---------- 6. 递归分组 ----------
#         function buildHierarchy(rows) {{
#           function rec(level,s,e){{
#             const nodes=[];
#             if(level>=rows[0].length) return nodes;
#             let i=s;
#             while(i<=e){{
#               const val=rows[i][level];
#               let j=i+1; while(j<=e && rows[j][level]===val) j++;
#               const node={{start:i,end:j-1,label:val,children:[]}};
#               node.children=rec(level+1,i,j-1);
#               nodes.push(node); i=j;
#             }}
#             return nodes;
#           }}
#           return rec(0,0,rows.length-1);
#         }}
#         const root = buildHierarchy(sortedRows);

#         function collectGroups(root,depth){{
#           const out=Array.from({{length:depth}},()=>[]);
#           function dfs(nodes,l){{for(const n of nodes){{out[l].push({{start:n.start,end:n.end,label:n.label}}); if(n.children.length) dfs(n.children,l+1);}}}}
#           dfs(root,0);
#           return out;
#         }}
#         const groupsPerLevel = collectGroups(root, maxDepth);

#         // ---------- 7. 分割线 ----------
#         function splitLines(){{
#           const arr=Array(uniqLabels.length+1).fill(null);
#           arr[0]=arr[uniqLabels.length]=maxDepth;
#           function mark(p,l){{if(arr[p]===null || l<arr[p]) arr[p]=l;}}
#           function walk(nodes,l){{
#             for(const n of nodes){{
#               mark(n.start,l); mark(n.end+1,l);
#               if(n.children.length) walk(n.children,l+1);
#               else for(let p=n.start;p<n.end;p++) mark(p+1,maxDepth);
#             }}
#           }}
#           walk(root,0);
#           return arr;
#         }}
#         const splitAt = splitLines();

#         // ---------- 8. ECharts ----------
#         const chart = echarts.init(document.getElementById('{chart_id}'));
#         const palette = ['#5470c6','#91cc75','#fac858','#ee6666','#73c0de','#3ba272','#fc8452','#9a60b4','#5ab1ef'];
#         const series = weeks.map((wk,i)=>({{
#           name:wk, type:'bar', data:sortedMerged[i],
#           itemStyle:{{color:palette[i%palette.length]}}, barMaxWidth:28
#         }}));

#         const reserved = maxDepth*gapBetweenAxes + extraBottomPadding;
#         const grid = {{...gridBase, bottom:gridBase.bottom+reserved}};

#         chart.setOption({{
#           title:{{text:raw.title, left:'center'}},
#           tooltip:{{trigger:'axis', axisPointer:{{type:'shadow'}},
#             formatter:p=>{{
#               if(!p||!p.length) return '';
#               const idx=p[0].dataIndex;
#               const lbl=uniqLabels[idx];
#               let html=`<b>${{lbl}}</b><br/>`;
#               p.forEach(o=>html+=`<span style="display:inline-block;width:10px;height:10px;background:${{o.color}};margin-right:6px;border-radius:2px;"></span>${{o.seriesName}}: ${{o.data}}<br/>`);
#               return html;
#             }}
#           }},
#           legend:{{top:36, right:grid.right-10}},
#           grid, xAxis:{{type:'category', data:uniqLabels.map((_,i)=>i+''), axisLine:{{show:false}}, axisTick:{{show:false}}, axisLabel:{{show:false}}}},
#           yAxis:{{type:'value', name:raw.yLabel}},
#           series
#         }});

#         // ---------- 9. Graphic ----------
#         function draw(){{
#           const w=chart.getWidth(), h=chart.getHeight();
#           const {{left,right,top,bottom}}=grid;
#           const innerW=w-left-right;
#           const catW=innerW/uniqLabels.length;
#           const gfx=[];

#           const bgColors=['#f5f5f5','#f9f9f9','#fcfcfc','#fefefe'];
#           const reservedTop=h-bottom;

#           // 背景+文字
#           for(let lvl=0;lvl<maxDepth;lvl++){{
#             const gs=groupsPerLevel[lvl];
#             const bg=bgColors[lvl%bgColors.length];
#             const y0=reservedTop+((maxDepth-1-lvl)*gapBetweenAxes)+6;
#             const hBand=gapBetweenAxes-8;
#             gs.forEach(g=>{{
#               if(!g.label) return;
#               const x=left+g.start*catW;
#               const w=(g.end-g.start+1)*catW;
#               gfx.push({{type:'rect',shape:{{x,y:y0,width:w,height:hBand}},style:{{fill:bg,stroke:'#e6e6e6',lineWidth:1}},z:1}});
#               gfx.push({{type:'text',style:{{text:g.label,x:x+w/2,y:y0+hBand/2,textAlign:'center',textVerticalAlign:'middle',fill:'#222',font:`${{labelFontSize}}px sans-serif`}},z:2}});
#             }});
#           }}

#           // 分割线
#           const lineTop=top-6;
#           splitAt.forEach((lvl,pos)=>{{
#             if(lvl===null) return;
#             const x=left+pos*catW;
#             const lineEnd=reservedTop+(maxDepth-lvl)*gapBetweenAxes;
#             gfx.push({{type:'line',shape:{{x1:x,y1:lineTop,x2:x,y2:lineEnd}},style:{{stroke:'#ccc',lineWidth:1}},z:3}});
#           }});

#           chart.setOption({{graphic:gfx}});
#         }}
#         draw();
#         window.addEventListener('resize',()=>{{
#           chart.resize(); setTimeout(draw,120);
#         }});
#       }})();
#       </script>
#     </body>
#     </html>
#     """
#     return html_template