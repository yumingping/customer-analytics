"""
MCP 工具: 智能图表生成 (chart_tools)
===================================
实现 readme.md 规定的 generate_visualization 工具。
- 根据数据特征自动选择最合适的图表类型
- 返回 ECharts 标准配置 JSON，前端可直接渲染
"""
import json
import pandas as pd
import numpy as np


# ==================== 图表类型推荐 ====================

def recommend_chart_type(data_description: dict) -> dict:
    """
    根据数据特征描述推荐最合适的图表类型。

    参数:
        data_description: {
            "columns": [...],
            "row_count": int,
            "column_types": {"col1": "categorical"|"numeric"|"datetime", ...},
            "purpose": "comparison"|"distribution"|"composition"|"relationship"|""  (可选)
        }

    返回: {"recommended": "bar", "alternatives": [...], "reason": "..."}
    """
    columns = data_description.get("columns", [])
    col_types = data_description.get("column_types", {})
    purpose = data_description.get("purpose", "")
    row_count = data_description.get("row_count", 0)

    # 统计列类型
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in col_types.items() if t == "datetime"]

    # 决策逻辑
    if purpose == "composition":
        if len(cat_cols) == 1 and len(numeric_cols) == 1 and row_count <= 15:
            recommended = "treemap"
            reason = "组成分析且类别较多，矩形树图比饼图更省空间"
        else:
            recommended = "pie"
            reason = "用途为组成分析，饼图最适合展示各部分占比"
    elif purpose == "relationship":
        if len(numeric_cols) >= 2:
            recommended = "scatter"
            reason = "用途为关系分析，散点图最适合展示数值关系"
        else:
            recommended = "bar"
            reason = "数据维度不足，降级为柱状图"
    elif purpose == "distribution":
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1 and row_count >= 3:
            recommended = "boxplot"
            reason = "用途为分布分析，箱线图展示分组分布与异常值"
        elif len(numeric_cols) >= 1:
            recommended = "bar"  # ECharts 用 bar 模拟直方图
            reason = "用途为分布分析，柱状图展示数据分布"
        else:
            recommended = "pie"
            reason = "分类数据，用饼图展示分布"
    elif purpose == "trend":
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            recommended = "area_line"
            reason = "时序趋势分析，面积折线图更直观"
        else:
            recommended = "line"
            reason = "趋势分析，折线图展示变化"
    elif purpose == "ranking":
        if len(cat_cols) == 1 and len(numeric_cols) == 1 and row_count >= 5:
            recommended = "pareto"
            reason = "排名分析，帕累托图展示关键少数"
        else:
            recommended = "bar"
            reason = "排名分析，柱状图展示大小"
    elif purpose == "funnel":
        recommended = "funnel"
        reason = "漏斗/阶段转化分析，漏斗图最适合"
    elif len(cat_cols) >= 1 and len(numeric_cols) >= 2:
        if row_count <= 12:
            recommended = "stacked_bar"
            reason = f"分类({cat_cols[0]})配多个数值，堆叠柱状图展示构成"
        else:
            recommended = "bar"
            reason = "多数值变量，分组柱状图更清晰"
    elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        if len(cat_cols) == 1 and len(numeric_cols) == 1:
            if row_count <= 10:
                recommended = "pie"
                reason = f"单一分类变量({cat_cols[0]})配数值，饼图最佳展示"
            else:
                recommended = "bar"
                reason = f"分类较多({row_count}行)，柱状图比饼图更清晰"
        else:
            recommended = "bar"
            reason = "多分类或多数值变量，柱状图最通用"
    elif len(numeric_cols) >= 2:
        recommended = "scatter"
        reason = f"多个数值变量({numeric_cols[:3]})，散点图展示关系"
    elif len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        recommended = "line"
        reason = f"时序变量({datetime_cols[0]})，折线图展示趋势"
    elif len(cat_cols) >= 2:
        recommended = "bar"
        reason = "多分类交叉，柱状图最清晰"
    else:
        recommended = "bar"
        reason = "通用降级方案"

    alternatives = {
        "bar": ["line", "pie"],
        "pie": ["bar", "treemap"],
        "line": ["bar", "scatter", "area_line"],
        "scatter": ["bar", "line"],
        "radar": ["bar"],
        "heatmap": ["scatter"],
        "funnel": ["bar", "pareto"],
        "boxplot": ["bar"],
        "gauge": ["bar"],
        "stacked_bar": ["bar", "line"],
        "area_line": ["line", "bar"],
        "treemap": ["pie", "bar"],
        "pareto": ["bar", "funnel"],
    }

    return {
        "recommended": recommended,
        "alternatives": alternatives.get(recommended, ["bar"]),
        "reason": reason,
    }


# ==================== ECharts 配置生成 ====================

def generate_visualization(data_json: str, chart_type: str = "auto",
                           title: str = "", purpose: str = "") -> dict:
    """
    根据数据生成 ECharts 配置 JSON，前端可直接渲染。
    （readme.md 第6节 MCP 工具规范）

    参数:
        data_json:   数据 JSON，格式 [{"col1": val, ...}, ...]
        chart_type:  图表类型 "auto"|"bar"|"pie"|"line"|"scatter"|"radar"|"heatmap"|
                     "funnel"|"boxplot"|"gauge"|"stacked_bar"|"area_line"|"treemap"|"pareto"
        title:       图表标题
        purpose:     分析目的 "comparison"|"distribution"|"composition"|"relationship"|"trend"|"ranking"|""

    返回:
        {
            "success": bool,
            "chart_type": str,
            "echarts_option": {...},  # ECharts 标准配置
            "recommendation": str,
        }
    """
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": f"数据解析失败: {e}"}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    # 分析列类型
    col_types = {}
    for col in df.columns:
        if df[col].dtype in ("int64", "float64", "int32", "float32"):
            # 检查是否为分类（唯一值少）
            unique_count = df[col].nunique()
            if unique_count <= 10 and unique_count < len(df) * 0.1:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numeric"
        elif "date" in col.lower() or "time" in col.lower():
            col_types[col] = "datetime"
        else:
            col_types[col] = "categorical"

    # 自动推荐图表类型
    if chart_type == "auto":
        desc = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "column_types": col_types,
            "purpose": purpose,
        }
        recommendation = recommend_chart_type(desc)
        chart_type = recommendation["recommended"]
    else:
        recommendation = {"recommended": chart_type, "reason": "用户指定"}

    # 生成 ECharts 配置
    builders = {
        "bar": _build_bar_option,
        "pie": _build_pie_option,
        "line": _build_line_option,
        "scatter": _build_scatter_option,
        "radar": _build_radar_option,
        "heatmap": _build_heatmap_option,
        "funnel": _build_funnel_option,
        "boxplot": _build_boxplot_option,
        "gauge": _build_gauge_option,
        "stacked_bar": _build_stacked_bar_option,
        "area_line": _build_area_line_option,
        "treemap": _build_treemap_option,
        "pareto": _build_pareto_option,
        "map": _build_map_option,
        "scatter3d": _build_scatter3d_option,
        "sankey": _build_sankey_option,
    }

    builder = builders.get(chart_type, _build_bar_option)
    try:
        echarts_option = builder(df, col_types, title)
    except Exception as e:
        return {
            "success": False,
            "error": f"图表生成失败: {e}",
            "chart_type": chart_type,
        }

    # 确保 data_json 被保留，供前端切换图表类型时使用
    raw_json = data_json if isinstance(data_json, str) else json.dumps(data, ensure_ascii=False)

    return {
        "success": True,
        "chart_type": chart_type,
        "echarts_option": echarts_option,
        "recommendation": recommendation.get("reason", ""),
        "data_json": raw_json,
        "title": title,
    }


# ==================== 各图表类型构造器 ====================

def _build_bar_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """柱状图"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if not cat_cols:
        cat_col = df.columns[0]
    else:
        cat_col = cat_cols[0]

    categories = df[cat_col].astype(str).tolist()
    series = []

    if numeric_cols:
        for ncol in numeric_cols[:5]:  # 最多5个系列
            series.append({
                "name": ncol,
                "type": "bar",
                "data": [float(v) if pd.notna(v) else 0 for v in df[ncol].tolist()],
            })
    else:
        # 对分类变量计数
        counts = df[cat_col].value_counts()
        categories = counts.index.astype(str).tolist()
        series.append({
            "name": "数量",
            "type": "bar",
            "data": [int(v) for v in counts.values],
        })

    return {
        "title": {"text": title or "柱状图", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": categories, "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "value"},
        "series": series,
        "grid": {"bottom": 100},
    }


def _build_pie_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """饼图"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if cat_cols and numeric_cols:
        names = df[cat_cols[0]].astype(str).tolist()
        values = [float(v) if pd.notna(v) else 0 for v in df[numeric_cols[0]].tolist()]
    elif cat_cols:
        counts = df[cat_cols[0]].value_counts()
        names = counts.index.astype(str).tolist()
        values = [int(v) for v in counts.values]
    else:
        names = df.iloc[:, 0].astype(str).tolist()
        values = [float(v) if pd.notna(v) else 0 for v in df.iloc[:, 1].tolist()]

    data = [{"name": n, "value": v} for n, v in zip(names, values)]

    return {
        "title": {"text": title or "饼图", "left": "center"},
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
        "legend": {"orient": "vertical", "left": "left"},
        "series": [{
            "name": title or "数据",
            "type": "pie",
            "radius": ["40%", "70%"],
            "data": data,
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
            "label": {"show": True, "formatter": "{b}: {d}%"},
        }],
    }


def _build_line_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """折线图"""
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "datetime")]
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    x_col = cat_cols[0] if cat_cols else df.columns[0]
    x_data = df[x_col].astype(str).tolist()
    series = []

    for ncol in numeric_cols[:5]:
        series.append({
            "name": ncol,
            "type": "line",
            "data": [float(v) if pd.notna(v) else None for v in df[ncol].tolist()],
            "smooth": True,
        })

    return {
        "title": {"text": title or "折线图", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x_data},
        "yAxis": {"type": "value"},
        "series": series,
    }


def _build_scatter_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """散点图：至少需要两个数值列，否则降级为柱状图"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    # 散点图要求 X、Y 轴都是数值，不足 2 个数值列时降级
    if len(numeric_cols) < 2:
        return _build_bar_option(df, col_types, title)

    x_col, y_col = numeric_cols[0], numeric_cols[1]

    # 按分类分组着色
    if cat_cols:
        group_col = cat_cols[0]
        groups = df[group_col].unique()
        series = []
        for g in groups:
            mask = df[group_col] == g
            series.append({
                "name": str(g),
                "type": "scatter",
                "data": [
                    [float(df.loc[i, x_col]), float(df.loc[i, y_col])]
                    for i in df[mask].index
                ],
            })
    else:
        series = [{
            "name": "数据点",
            "type": "scatter",
            "data": [
                [float(df.loc[i, x_col]), float(df.loc[i, y_col])]
                for i in df.index
            ],
        }]

    return {
        "title": {"text": title or "散点图", "left": "center"},
        "tooltip": {"trigger": "item", "formatter": f"{x_col}: {{c[0]}}<br/>{y_col}: {{c[1]}}"},
        "xAxis": {"name": x_col, "type": "value"},
        "yAxis": {"name": y_col, "type": "value"},
        "series": series,
    }


def _build_radar_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """雷达图"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    indicators = []
    for col in numeric_cols[:8]:
        max_val = float(df[col].max()) if not df[col].empty else 100
        indicators.append({"name": col, "max": round(max_val * 1.2, 1)})

    if cat_cols:
        group_col = cat_cols[0]
        groups = df[group_col].unique()
        series = []
        for g in groups:
            mask = df[group_col] == g
            vals = [float(df.loc[mask, col].mean()) for col in numeric_cols[:8]]
            series.append({"name": str(g), "type": "radar", "data": [{"value": vals, "name": str(g)}]})
    else:
        vals = [float(df[col].mean()) for col in numeric_cols[:8]]
        series = [{"name": title or "数据", "type": "radar", "data": [{"value": vals, "name": title or "均值"}]}]

    return {
        "title": {"text": title or "雷达图", "left": "center"},
        "tooltip": {},
        "radar": {"indicator": indicators},
        "series": series,
    }


def _build_heatmap_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """热力图（相关性矩阵）"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    if len(numeric_cols) < 2:
        return _build_bar_option(df, col_types, title)

    corr = df[numeric_cols].corr().round(3)

    heat_data = []
    for i, row_name in enumerate(corr.columns):
        for j, col_name in enumerate(corr.columns):
            heat_data.append([i, j, float(corr.loc[row_name, col_name])])

    return {
        "title": {"text": title or "相关性热力图", "left": "center"},
        "tooltip": {"position": "top"},
        "xAxis": {"type": "category", "data": corr.columns.tolist(), "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "category", "data": corr.columns.tolist()},
        "visualMap": {"min": -1, "max": 1, "calculable": True,
                       "inRange": {"color": ["#313695", "#4575b4", "#74add1", "#abd9e9",
                                              "#fee090", "#fdae61", "#f46d43", "#d73027"]}},
        "series": [{
            "name": "相关性",
            "type": "heatmap",
            "data": heat_data,
            "label": {"show": True, "fontSize": 10},
        }],
        "grid": {"bottom": 120},
    }


def _build_funnel_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """漏斗图：需要一列分类（stage）和一列数值（value）"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if cat_cols and numeric_cols:
        name_col, value_col = cat_cols[0], numeric_cols[0]
    elif len(df.columns) >= 2:
        name_col, value_col = df.columns[0], df.columns[-1]
    else:
        return _build_bar_option(df, col_types, title)

    data = [
        {"name": str(n), "value": float(v) if pd.notna(v) else 0}
        for n, v in zip(df[name_col], df[value_col])
    ]
    data = sorted(data, key=lambda x: x["value"], reverse=True)

    return {
        "title": {"text": title or "漏斗图", "left": "center"},
        "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
        "series": [{
            "name": title or "漏斗",
            "type": "funnel",
            "left": "10%", "top": 60, "bottom": 60,
            "width": "80%",
            "min": 0,
            "max": max([d["value"] for d in data]) if data else 100,
            "sort": "descending",
            "gap": 2,
            "label": {"show": True, "position": "inside", "formatter": "{b}: {c}"},
            "data": data,
        }],
    }


def _build_boxplot_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """箱线图：需要一列分类（x轴）和至少一列数值"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if not numeric_cols:
        return _build_bar_option(df, col_types, title)

    if cat_cols:
        group_col = cat_cols[0]
        groups = sorted(df[group_col].unique())
    else:
        group_col = "数值"
        groups = [group_col]

    series_data = []
    for g in groups:
        if cat_cols:
            vals = df[df[group_col] == g][numeric_cols[0]].dropna().tolist()
        else:
            vals = df[numeric_cols[0]].dropna().tolist()
        series_data.append(vals)

    return {
        "title": {"text": title or "箱线图", "left": "center"},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "category", "data": [str(g) for g in groups]},
        "yAxis": {"type": "value"},
        "series": [{
            "name": numeric_cols[0],
            "type": "boxplot",
            "data": series_data,
        }],
        "grid": {"bottom": 80},
    }


def _build_gauge_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """仪表盘：取第一个数值列的平均值"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    if not numeric_cols:
        return _build_bar_option(df, col_types, title)

    value = float(df[numeric_cols[0]].mean())
    max_val = float(df[numeric_cols[0]].max())
    if max_val <= value:
        max_val = value * 1.2

    return {
        "title": {"text": title or f"{numeric_cols[0]} 仪表盘", "left": "center"},
        "series": [{
            "type": "gauge",
            "startAngle": 180, "endAngle": 0,
            "min": 0, "max": round(max_val, 2),
            "splitNumber": 5,
            "itemStyle": {"color": "#6366f1"},
            "progress": {"show": True, "width": 18},
            "pointer": {"show": True},
            "axisLine": {"lineStyle": {"width": 18}},
            "axisTick": {"show": True},
            "splitLine": {"length": 15, "lineStyle": {"width": 2}},
            "axisLabel": {"distance": 25, "fontSize": 12},
            "anchor": {"show": True, "size": 25, "itemStyle": {"borderWidth": 10}},
            "title": {"show": True},
            "detail": {"valueAnimation": True, "fontSize": 30, "offsetCenter": [0, "70%"]},
            "data": [{"value": round(value, 2), "name": numeric_cols[0]}],
        }],
    }


def _build_stacked_bar_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """堆叠柱状图：分类轴 + 多个数值系列"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if not cat_cols or len(numeric_cols) < 2:
        return _build_bar_option(df, col_types, title)

    x_col = cat_cols[0]
    categories = df[x_col].astype(str).unique().tolist()
    series = []

    for ncol in numeric_cols[:6]:
        data = []
        for cat in categories:
            vals = df[df[x_col] == cat][ncol]
            data.append(float(vals.mean()) if not vals.empty else 0)
        series.append({
            "name": ncol,
            "type": "bar",
            "stack": "total",
            "emphasis": {"focus": "series"},
            "data": data,
        })

    return {
        "title": {"text": title or "堆叠柱状图", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": "bottom"},
        "xAxis": {"type": "category", "data": categories},
        "yAxis": {"type": "value"},
        "series": series,
        "grid": {"bottom": 100},
    }


def _build_area_line_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """面积折线图：时序/分类轴 + 数值系列"""
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "datetime")]
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    x_col = cat_cols[0] if cat_cols else df.columns[0]
    x_data = df[x_col].astype(str).tolist()
    series = []

    for ncol in numeric_cols[:5]:
        series.append({
            "name": ncol,
            "type": "line",
            "smooth": True,
            "areaStyle": {"opacity": 0.3},
            "data": [float(v) if pd.notna(v) else None for v in df[ncol].tolist()],
        })

    return {
        "title": {"text": title or "面积折线图", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": "bottom"},
        "xAxis": {"type": "category", "boundaryGap": False, "data": x_data},
        "yAxis": {"type": "value"},
        "series": series,
        "grid": {"bottom": 100},
    }


def _build_treemap_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """矩形树图：分类 + 数值，铺平展示避免拥挤。超过 15 类时合并为"其他"。"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if cat_cols and numeric_cols:
        name_col, value_col = cat_cols[0], numeric_cols[0]
    elif len(df.columns) >= 2:
        name_col, value_col = df.columns[0], df.columns[-1]
    else:
        return _build_bar_option(df, col_types, title)

    # 聚合重复分类值，按值降序排列
    agg = df.groupby(name_col)[value_col].sum().sort_values(ascending=False)

    # 超过 15 类时，保留 Top 15，其余合并为"其他"，避免小矩形拥挤
    MAX_ITEMS = 15
    if len(agg) > MAX_ITEMS:
        top = agg.head(MAX_ITEMS)
        others_val = float(agg.iloc[MAX_ITEMS:].sum())
        data = [
            {"name": str(n), "value": float(v) if pd.notna(v) else 0}
            for n, v in top.items()
        ]
        if others_val > 0:
            data.append({"name": "其他", "value": others_val})
    else:
        data = [
            {"name": str(n), "value": float(v) if pd.notna(v) else 0}
            for n, v in agg.items()
        ]

    return {
        "title": {"text": title or "矩形树图", "left": "center", "top": 10},
        "tooltip": {"formatter": "{b}: {c}"},
        "series": [{
            "type": "treemap",
            "data": data,
            "roam": True,
            "nodeClick": "zoomToNode",
            "leafDepth": 1,
            "breadcrumb": {"show": True, "bottom": 5},
            "label": {
                "show": True,
                "formatter": "{b}\n{c}",
                "fontSize": 12,
                "color": "#333",
                "minAngle": 5,
            },
            "upperLabel": {"show": True, "height": 22},
            "itemStyle": {
                "borderColor": "#fff",
                "borderWidth": 2,
                "gapWidth": 3,
            },
            "levels": [
                {
                    "itemStyle": {"borderColor": "#d4d4d4", "borderWidth": 2, "gapWidth": 3},
                    "upperLabel": {"show": False},
                },
                {
                    "colorSaturation": [0.35, 0.5],
                    "itemStyle": {"borderColor": "#fff", "borderWidth": 1, "gapWidth": 2},
                    "upperLabel": {"show": True},
                },
            ],
        }],
        "grid": {"top": 50, "bottom": 40},
    }


def _build_pareto_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """帕累托图：分类 + 数值柱状 + 累计百分比折线"""
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if not cat_cols or not numeric_cols:
        return _build_bar_option(df, col_types, title)

    name_col, value_col = cat_cols[0], numeric_cols[0]
    agg = df.groupby(name_col)[value_col].sum().sort_values(ascending=False)
    total = agg.sum()
    cumulative = agg.cumsum() / total * 100

    return {
        "title": {"text": title or "帕累托图", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"top": "bottom"},
        "xAxis": {"type": "category", "data": agg.index.astype(str).tolist()},
        "yAxis": [
            {"type": "value", "name": value_col, "position": "left"},
            {"type": "value", "name": "累计百分比(%)", "position": "right", "max": 100},
        ],
        "series": [
            {
                "name": value_col,
                "type": "bar",
                "data": [round(float(v), 2) for v in agg.values],
                "itemStyle": {"borderRadius": [4, 4, 0, 0]},
            },
            {
                "name": "累计百分比",
                "type": "line",
                "yAxisIndex": 1,
                "data": [round(float(v), 2) for v in cumulative.values],
                "symbol": "circle",
            },
        ],
        "grid": {"bottom": 100},
    }


def _build_map_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """
    地理地图：需要地区名称列和数值列
    注意：前端需要加载 GeoJSON 地图数据
    """
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    if not cat_cols or not numeric_cols:
        return _build_bar_option(df, col_types, title)

    name_col = cat_cols[0]
    value_col = numeric_cols[0]

    # 聚合数据
    agg = df.groupby(name_col)[value_col].sum()
    data = [
        {"name": str(name), "value": float(val) if pd.notna(val) else 0}
        for name, val in agg.items()
    ]

    return {
        "title": {"text": title or "地理地图", "left": "center"},
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {c}"
        },
        "visualMap": {
            "min": 0,
            "max": max([d["value"] for d in data]) if data else 100,
            "left": "left",
            "top": "bottom",
            "text": ["高", "低"],
            "calculable": True,
            "inRange": {
                "color": ["#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027"]
            }
        },
        "series": [{
            "name": value_col,
            "type": "map",
            "map": "china",  # 需要前端注册地图
            "roam": True,
            "label": {
                "show": True,
                "fontSize": 10
            },
            "emphasis": {
                "label": {"show": True},
                "itemStyle": {"areaColor": "#ffd700"}
            },
            "data": data
        }]
    }


def _build_scatter3d_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """
    3D 散点图：需要至少三个数值列
    注意：需要 echarts-gl 扩展
    """
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    if len(numeric_cols) < 3:
        # 降级为普通散点图
        return _build_scatter_option(df, col_types, title)

    x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]

    # 构建 3D 数据点
    data = []
    for _, row in df.iterrows():
        x = float(row[x_col]) if pd.notna(row[x_col]) else 0
        y = float(row[y_col]) if pd.notna(row[y_col]) else 0
        z = float(row[z_col]) if pd.notna(row[z_col]) else 0
        data.append([x, y, z])

    return {
        "title": {"text": title or "3D 散点图", "left": "center"},
        "tooltip": {
            "formatter": lambda idx: f"{x_col}: {data[idx][0]}<br/>{y_col}: {data[idx][1]}<br/>{z_col}: {data[idx][2]}"
        },
        "xAxis3D": {
            "type": "value",
            "name": x_col
        },
        "yAxis3D": {
            "type": "value",
            "name": y_col
        },
        "zAxis3D": {
            "type": "value",
            "name": z_col
        },
        "grid3D": {
            "viewControl": {
                "projection": "perspective",
                "autoRotate": True
            },
            "light": {
                "main": {"intensity": 1.2, "shadow": True},
                "ambient": {"intensity": 0.3}
            }
        },
        "series": [{
            "type": "scatter3D",
            "data": data,
            "symbolSize": 8,
            "itemStyle": {
                "opacity": 0.8,
                "color": "#6366f1"
            },
            "emphasis": {
                "itemStyle": {
                    "color": "#f59e0b"
                }
            }
        }]
    }


def _build_sankey_option(df: pd.DataFrame, col_types: dict, title: str) -> dict:
    """
    桑基图：需要源节点列、目标节点列和流量值列
    用于展示流向关系
    """
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

    if len(cat_cols) < 2 or not numeric_cols:
        return _build_bar_option(df, col_types, title)

    source_col = cat_cols[0]
    target_col = cat_cols[1]
    value_col = numeric_cols[0]

    # 构建节点和链接
    nodes = set()
    links = []

    for _, row in df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        value = float(row[value_col]) if pd.notna(row[value_col]) else 0

        if value > 0:
            nodes.add(source)
            nodes.add(target)
            links.append({
                "source": source,
                "target": target,
                "value": value
            })

    # 合并相同源目标的链接
    link_dict = {}
    for link in links:
        key = (link["source"], link["target"])
        if key in link_dict:
            link_dict[key]["value"] += link["value"]
        else:
            link_dict[key] = link.copy()

    nodes_list = [{"name": name} for name in nodes]
    links_list = list(link_dict.values())

    return {
        "title": {"text": title or "桑基图", "left": "center"},
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {c}"
        },
        "series": [{
            "type": "sankey",
            "layout": "none",
            "emphasis": {
                "focus": "adjacency"
            },
            "nodeAlign": "left",
            "data": nodes_list,
            "links": links_list,
            "lineStyle": {
                "color": "gradient",
                "curveness": 0.5
            },
            "label": {
                "show": True,
                "fontSize": 12
            }
        }]
    }
