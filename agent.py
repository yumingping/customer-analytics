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

1. **get_database_schema** - 获取完整数据库结构（编写 SQL 前必须调用）
2. **get_table_schema** - 获取单张表的结构
3. **probe_distinct_values** - 嗅探字段的实际取值（构建 WHERE 条件前必须调用）
4. **execute_secure_sql** - 执行 SELECT 查询
5. **generate_visualization** - 生成图表（当用户要求可视化时**必须调用**）
6. **perform_dynamic_cluster** - 执行聚类分析
7. **analyze_cluster_profiles** - 分析聚类画像
8. **forecast_trend** - 趋势预测
9. **detect_anomalies** - 异常检测
10. **churn_risk_score** - 流失风险评估

**工作流程：**
1. 理解用户需求
2. 如需查询数据，先调用 get_database_schema 了解表结构
3. 构建 WHERE 条件前，调用 probe_distinct_values 确认字段取值
4. 调用 execute_secure_sql 执行查询
5. **如用户要求画图、可视化、图表、直方图、柱状图、饼图等，必须调用 generate_visualization 工具**
6. 用中文总结结果并回答用户

**重要规则：**
- 只使用数据库中存在的字段，不要编造字段名
- 数据库包含：客户基础信息、飞行行为、RFM 分析、聚类结果
- 如果不确定字段名，先查询 schema
- 用简洁清晰的中文回答
- **当用户提到"画图"、"图表"、"可视化"、"直方图"、"柱状图"、"饼图"等关键词时，必须调用 generate_visualization 工具，不要只返回文字描述**
"""


# ==================== Function Calling 模式 ====================

def agent_loop_function_calling(question: str, max_iterations: int = 10) -> Dict[str, Any]:
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
        {"role": "user", "content": question}
    ]
    
    steps = []
    iteration = 0
    
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
            
            # 调用工具
            print(f"[Agent] 调用工具: {tool_name}({tool_args})")
            tool_result = client.call_tool(tool_name, tool_args)
            
            # 记录步骤
            steps.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
                "success": "error" not in tool_result
            })
            
            # 添加工具结果到消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result, ensure_ascii=False)
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

def agent_loop_react(question: str, max_iterations: int = 8) -> Dict[str, Any]:
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

**使用格式：**
当你需要使用工具时，请按以下格式输出：
Thought: 思考下一步该做什么
Action: 工具名称
Action Input: {{"参数1": "值1", "参数2": "值2"}}

系统会返回工具执行结果，然后你继续思考。

当你有足够信息回答用户时，输出：
Thought: 我现在可以回答用户了
Final Answer: 你的回答

**重要：**
- 每次只调用一个工具
- Action Input 必须是合法的 JSON
- 不要编造工具名称或参数
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
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
            
            # 添加工具结果
            messages.append({
                "role": "assistant",
                "content": response_text
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {json.dumps(tool_result, ensure_ascii=False)}\n\n继续思考："
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

def run_agent(question: str) -> Dict[str, Any]:
    """
    根据 LLM 能力自动选择 Agent 模式
    
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
        return agent_loop_function_calling(question)
    elif config.LLM_PROVIDER == "ollama":
        # gemma3:4b 等，使用 ReAct 模式
        print("[Agent] 使用 ReAct 模式")
        return agent_loop_react(question)
    else:
        return {
            "answer": "未配置 LLM 提供商",
            "steps": [],
            "mode": "error"
        }
