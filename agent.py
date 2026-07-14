"""
Agent 核心模块
实现基于 MCP 的智能体循环，支持 Function Calling 和 ReAct 两种模式
"""
import json
import requests
from typing import Dict, Any, List, Optional
import config
from mcp_client import get_mcp_client, is_mcp_available


# ==================== Agent System Prompt ====================

AGENT_SYSTEM_PROMPT = """你是航空公司客户数据分析平台的智能助手。

你可以使用以下工具来帮助用户：

1. **get_database_schema** - 获取完整数据库结构（**编写任何 SQL 前必须先调用此工具**）
2. **get_table_schema** - 获取单张表的结构
3. **probe_distinct_values** - 嗅探字段的实际取值（构建 WHERE 条件前必须调用）
4. **execute_secure_sql** - 执行 SELECT 查询（仅用于小数据量查询，大数据建模请用下方的专用工具）
5. **generate_visualization** - 生成图表（当用户要求可视化时**必须调用**）
6. **perform_dynamic_cluster** - 执行聚类分析
7. **analyze_cluster_profiles** - 分析聚类画像
8. **forecast_trend** - 趋势预测。支持 sql_query/table_name 参数，工具内部直连数据库加载全量数据
9. **detect_anomalies** - 异常检测。支持 sql_query/table_name 参数，工具内部直连数据库加载全量数据
10. **churn_risk_score** - 流失风险评分。支持 sql_query/table_name 参数，工具内部直连数据库加载全量数据

**核心原则：先查 Schema，再写 SQL**

数据库有三张核心表（通过 member_no 关联）：
- **customer_base**：客户基础信息（age, gender, ffp_tier, avg_discount 等）
- **customer_flight_summary**：飞行行为汇总（recency, flight_count, seg_km_sum, bp_sum, last_flight）
- **customer_analytics**：分析结果（r_score, f_score, m_score, rfm_total, cluster, value_label）
- **customer_rfm**：三表 JOIN 宽表（兼容旧查询，含所有字段）

**【重要】Compute-over-Data 原则 — 避免数据截断：**

churn_risk_score、forecast_trend、detect_anomalies 这三个工具都支持 sql_query 和 table_name 参数。
**绝对不要**先用 execute_secure_sql 拉取全量数据，再以 data_json 传给建模工具！
这样会导致数据被截断到 30 行，建模结果毫无意义。

正确做法：
- 直接调用 churn_risk_score，用 sql_query 参数传入你的 SQL 语句
- 工具内部会直连数据库加载全量数据（62,987 行），不经过 LLM 上下文
- 工具只返回聚合摘要（如分布、均值），数据量极小，不会被截断
- 示例: churn_risk_score(sql_query="SELECT member_no, recency, flight_count, seg_km_sum FROM customer_flight_summary")

**关键规则：**
- 编写 SQL 前必须调用 get_database_schema，确认实际存在的表和列（以 _actual_columns 为准）
- 返回结果中的 _actual_columns 是数据库真实的列名，必须以它为准
- 如果 _missing_columns 不为空，说明 Schema 定义的某些列在数据库中不存在，不要使用它们
- 构建 WHERE 条件前，调用 probe_distinct_values 确认字段的真实取值
- 只使用数据库中真实存在的字段，禁止编造字段名
- customer_analytics 的 value_label 取值：'高价值客户', '中价值客户', '低价值客户', '流失客户'
- 如果 SQL 执行失败，仔细阅读 error_hint 中的修复建议，修正后重试

**工作流程：**
1. 理解用户需求
2. 调用 get_database_schema 了解实际表结构
3. 如需条件筛选，调用 probe_distinct_values 确认取值
4. 如需建模分析（流失、预测、异常），直接将 SQL 以 sql_query 参数传给建模工具
5. 如需简单查询，调用 execute_secure_sql 执行
6. **如用户要求画图、可视化、图表、直方图、柱状图、饼图等，必须调用 generate_visualization 工具**
7. 用中文总结结果并回答用户

**【重要】输出要求 — 每次获取数据后必须提供分析总结：**
- 不论数据量多少，必须在最终回答中包含：
  a) 数据概览：总共多少条记录、涉及哪些维度
  b) 关键发现：数值的统计特征（均值、范围、分布特点）
  c) 业务解读：这些数据说明了什么业务现象
- 如果工具返回的数据被截断（含 _truncated: true），只需要基于样本数据做分析，并说明"以上为抽样结果"
- 严禁只输出"查询完成"、"执行完毕"等无意义短语，必须输出有业务价值的分析内容
"""


# ==================== 工具结果截断（防 LLM 上下文溢出） ====================

def _truncate_for_llm(tool_name: str, result: dict) -> dict:
    """
    截断大工具结果，防止 LLM 上下文被撑爆导致分析失败。
    对 SQL 查询结果只保留摘要 + 前 30 行样本，其他工具保留前 50 行。
    """
    if not isinstance(result, dict):
        return result

    MAX_SAMPLE_ROWS = 30
    MAX_JSON_CHARS = 8000

    # 复制一份避免修改原始数据（steps 记录需要完整结果）
    truncated = dict(result)

    # 处理 SQL 查询结果
    data_key = None
    for key in ("data", "anomalies", "data_with_labels", "historical", "forecast"):
        if key in truncated and isinstance(truncated[key], list):
            data_key = key
            break

    if data_key:
        full_data = truncated[data_key]
        total = len(full_data)

        if total > MAX_SAMPLE_ROWS:
            truncated[data_key] = full_data[:MAX_SAMPLE_ROWS]
            truncated["_truncated"] = True
            truncated["_total_rows"] = total
            truncated["_shown_rows"] = MAX_SAMPLE_ROWS
            truncated["_note"] = f"数据量过大({total}行)，仅展示前{MAX_SAMPLE_ROWS}行样本"

    # 二次保护：JSON 序列化后仍超长则再次截断
    try:
        serialized = json.dumps(truncated, ensure_ascii=False, default=str)
        if len(serialized) > MAX_JSON_CHARS:
            # 极端情况：逐行删除直到符合长度
            if data_key and isinstance(truncated.get(data_key), list):
                while len(truncated[data_key]) > 5:
                    truncated[data_key] = truncated[data_key][:len(truncated[data_key]) - 5]
                    truncated["_shown_rows"] = len(truncated[data_key])
                    if len(json.dumps(truncated, ensure_ascii=False, default=str)) <= MAX_JSON_CHARS:
                        break
                truncated["_note"] = f"数据量过大({total}行)，已大幅截断至{len(truncated[data_key])}行"
    except Exception:
        pass

    return truncated


# ==================== Function Calling 模式 ====================

def agent_loop_function_calling(question: str, history: List[Dict] = None, max_iterations: int = 20) -> Dict[str, Any]:
    """
    使用 Function Calling 的 Agent Loop（适用于 DeepSeek、GPT 等）
    
    Args:
        question: 用户问题
        max_iterations: 最大迭代次数
        
    Returns:
        {
            "answer": str,           # 最终回答
            "steps": List[Dict],     # 工具调用步骤
            "mode": "agent_fc"       # 运行模式
        }
    """
    client = get_mcp_client()
    
    # 获取工具定义
    tools_def = client.list_tools()
    if not tools_def:
        return {
            "answer": "MCP Server 不可用，无法获取工具定义",
            "steps": [],
            "mode": "error"
        }
    
    # 构建 OpenAI 格式的工具定义
    tools = []
    for tool in tools_def:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {})
            }
        })
    
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    ]

    # 注入历史对话上下文（让 LLM 理解上文指代，如"上面"、"上文"、"根据上文"等）
    if history:
        # 限制历史长度，避免 token 过多（最多保留最近 6 轮）
        recent_history = history[-12:]  # 6 轮对话
        for h in recent_history:
            role = h.get("role", "")
            content = h.get("content", "")
            if not content:
                continue
            # 截断过长的内容
            if len(content) > 800:
                content = content[:800] + "..."
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    
    steps = []
    iteration = 0
    tool_call_counts = {}  # 记录每个工具的调用次数
    MAX_TOOL_CALLS = 3  # 每个工具最多调用3次
    
    while iteration < max_iterations:
        iteration += 1
        
        # 调用 LLM
        try:
            response = _call_llm_with_tools(messages, tools)
        except Exception as e:
            return {
                "answer": f"LLM 调用失败: {str(e)}",
                "steps": steps,
                "mode": "error"
            }
        
        # 解析响应
        message = response.get("choices", [{}])[0].get("message", {})
        
        # 检查是否有工具调用
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            # 没有工具调用，返回最终回答
            answer = message.get("content", "")
            return {
                "answer": answer,
                "steps": steps,
                "mode": "agent_fc"
            }
        
        # 处理工具调用
        messages.append(message)  # 添加 assistant 消息
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]
            
            # 检查工具调用次数限制
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            if tool_call_counts[tool_name] > MAX_TOOL_CALLS:
                print(f"[Agent] 工具 {tool_name} 已达到最大调用次数 {MAX_TOOL_CALLS}，跳过")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": f"工具 {tool_name} 已达到最大调用次数 {MAX_TOOL_CALLS}"}, ensure_ascii=False)
                })
                continue
            
            # 调用工具
            print(f"[Agent] 调用工具: {tool_name}({tool_args})")
            tool_result = client.call_tool(tool_name, tool_args)
            
            # 记录步骤（保留完整结果供前端展示，不受截断影响）
            steps.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
                "success": "error" not in tool_result
            })
            
            # 传给 LLM 的结果需要截断，防止大数据撑爆上下文
            llm_result = _truncate_for_llm(tool_name, tool_result)
            if llm_result.get("_truncated"):
                print(f"[Agent] 工具 {tool_name} 结果已截断: {llm_result.get('_total_rows')} → {llm_result.get('_shown_rows')} 行")
            
            # 添加工具结果到消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(llm_result, ensure_ascii=False)
            })
    
    return {
        "answer": "达到最大迭代次数，任务未完成",
        "steps": steps,
        "mode": "agent_fc"
    }


def _call_llm_with_tools(messages: List[Dict], tools: List[Dict]) -> Dict:
    """调用支持 Function Calling 的 LLM"""
    if config.LLM_PROVIDER == "openai_compatible" and config.OPENAI_API_KEY:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENAI_API_KEY}"
        }
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }
        response = requests.post(
            config.OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        return response.json()
    else:
        raise ValueError("当前配置不支持 Function Calling 模式")


# ==================== ReAct 模式（降级方案） ====================

def agent_loop_react(question: str, history: List[Dict] = None, max_iterations: int = 15) -> Dict[str, Any]:
    """
    使用 ReAct 提示词的 Agent Loop（适用于 gemma3:4b 等不支持 Function Calling 的模型）
    
    Args:
        question: 用户问题
        max_iterations: 最大迭代次数
        
    Returns:
        {
            "answer": str,
            "steps": List[Dict],
            "mode": "agent_react"
        }
    """
    client = get_mcp_client()
    tools_def = client.list_tools()
    
    # 构建工具描述
    tools_desc = []
    for tool in tools_def:
        tools_desc.append(f"- {tool['name']}: {tool.get('description', '')}")
    tools_str = "\n".join(tools_desc)
    
    system_prompt = f"""你是一个智能助手，可以使用以下工具来帮助用户：

{tools_str}

**核心原则：先查 Schema，再写 SQL**

数据库有三张核心表（通过 member_no 关联）：
- customer_base：客户基础信息（age, gender, ffp_tier, avg_discount）
- customer_flight_summary：飞行行为汇总（recency, flight_count, seg_km_sum, bp_sum, last_flight）
- customer_analytics：分析结果（r_score, f_score, m_score, rfm_total, cluster, value_label）
- customer_rfm：三表 JOIN 宽表（兼容旧查询）

**【重要】Compute-over-Data — 避免数据截断：**
churn_risk_score、forecast_trend、detect_anomalies 都支持 sql_query 参数。
不要先用 execute_secure_sql 拉数据再传给建模工具（会被截断到30行）。
直接调用建模工具，用 sql_query 参数传入 SQL，工具内部直连数据库加载全量数据。

**使用格式：**
当你需要使用工具时，请按以下格式输出：
Thought: 思考下一步该做什么
Action: 工具名称
Action Input: {{"参数1": "值1", "参数2": "值2"}}

系统会返回工具执行结果，然后你继续思考。

当你有足够信息回答用户时，输出：
Thought: 我现在可以回答用户了
Final Answer: 你的回答（必须包含数据分析总结，禁止只输出"查询完成"）

**重要规则：**
- 编写 SQL 前必须先调用 get_database_schema 确认实际列名（以 _actual_columns 为准）
- 如果 _missing_columns 不为空，说明某些列在数据库中不存在
- 构建 WHERE 条件前，调用 probe_distinct_values 确认取值
- 每次只调用一个工具
- Action Input 必须是合法的 JSON
- 不要编造工具名称或参数
- 如果 SQL 执行失败，根据 error_hint 的建议修正后重试
- 获取数据后必须提供分析总结（数据概览、关键发现、业务解读）
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # 注入历史对话上下文
    if history:
        recent_history = history[-12:]  # 最多保留最近 6 轮
        for h in recent_history:
            role = h.get("role", "")
            content = h.get("content", "")
            if not content:
                continue
            if len(content) > 600:
                content = content[:600] + "..."
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    
    steps = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # 调用 LLM
        try:
            response_text = _call_llm_react(messages)
        except Exception as e:
            return {
                "answer": f"LLM 调用失败: {str(e)}",
                "steps": steps,
                "mode": "error"
            }
        
        # 解析响应
        if "Final Answer:" in response_text:
            answer = response_text.split("Final Answer:")[-1].strip()
            return {
                "answer": answer,
                "steps": steps,
                "mode": "agent_react"
            }
        
        # 提取 Action
        if "Action:" in response_text and "Action Input:" in response_text:
            action_line = [line for line in response_text.split("\n") if line.startswith("Action:")][0]
            tool_name = action_line.replace("Action:", "").strip()
            
            input_line = [line for line in response_text.split("\n") if line.startswith("Action Input:")][0]
            input_json = input_line.replace("Action Input:", "").strip()
            
            try:
                tool_args = json.loads(input_json)
            except json.JSONDecodeError:
                tool_args = {}
            
            # 调用工具
            print(f"[Agent ReAct] 调用工具: {tool_name}({tool_args})")
            tool_result = client.call_tool(tool_name, tool_args)
            
            steps.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
                "success": "error" not in tool_result
            })
            
            # 传给 LLM 的结果截断
            llm_result = _truncate_for_llm(tool_name, tool_result)
            if llm_result.get("_truncated"):
                print(f"[Agent ReAct] 工具 {tool_name} 结果已截断: {llm_result.get('_total_rows')} → {llm_result.get('_shown_rows')} 行")
            
            # 添加工具结果
            messages.append({
                "role": "assistant",
                "content": response_text
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {json.dumps(llm_result, ensure_ascii=False)}\n\n继续思考："
            })
        else:
            # 无法解析，直接返回
            return {
                "answer": response_text,
                "steps": steps,
                "mode": "agent_react"
            }
    
    return {
        "answer": "达到最大迭代次数，任务未完成",
        "steps": steps,
        "mode": "agent_react"
    }


def _call_llm_react(messages: List[Dict]) -> str:
    """调用不支持 Function Calling 的 LLM（如 gemma3:4b）"""
    # 构建 prompt
    prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    
    if config.LLM_PROVIDER == "ollama":
        response = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )
        return response.json().get("response", "")
    else:
        raise ValueError("当前配置不支持 ReAct 模式")


# ==================== 智能选择模式 ====================

def run_agent(question: str, history: List[Dict] = None) -> Dict[str, Any]:
    """
    根据 LLM 能力自动选择 Agent 模式

    Args:
        question: 当前用户问题
        history: 历史对话 [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        {
            "answer": str,
            "steps": List[Dict],
            "mode": str
        }
    """
    # 检查 MCP Server 是否可用
    if not is_mcp_available():
        return {
            "answer": "MCP Server 不可用，请确保 api.py 正在运行",
            "steps": [],
            "mode": "error"
        }

    # 根据配置选择模式
    if config.LLM_PROVIDER == "openai_compatible" and config.OPENAI_API_KEY:
        # 支持 Function Calling 的模型
        print("[Agent] 使用 Function Calling 模式")
        return agent_loop_function_calling(question, history=history)
    elif config.LLM_PROVIDER == "ollama":
        # gemma3:4b 等，使用 ReAct 模式
        print("[Agent] 使用 ReAct 模式")
        return agent_loop_react(question, history=history)
    else:
        return {
            "answer": "未配置 LLM 提供商",
            "steps": [],
            "mode": "error"
        }
