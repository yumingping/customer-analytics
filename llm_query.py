"""大模型辅助查询模块 (Ollama + Gemma:4b)
将自然语言问题转换为 SQL，执行查询并返回结果
"""
import re
import pandas as pd
import requests
import json
from sqlalchemy import create_engine, text
import config


def ask_gemma(prompt: str) -> str:
    """调用 Ollama API 获取模型回复"""
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": config.OLLAMA_OPTIONS,
    }
    try:
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"无法连接到 Ollama 服务 ({config.OLLAMA_URL})。\n"
            f"请确保 Ollama 已启动: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama 请求超时，请检查模型是否已加载")
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"解析 Ollama 响应失败: {e}")


def build_system_prompt() -> str:
    """构建 SQL 生成任务提示词"""
    return f"""
你是一个SQL专家。数据库表 `{config.TABLE_NAME}` 的结构如下：

字段说明：
- member_no (VARCHAR, 主键) - 会员编号
- age (INT) - 年龄
- gender (VARCHAR) - 性别
- ffp_tier (INT) - 会员等级
- recency (INT) - 距离观察期末的天数（越小表示最近越活跃）
- frequency (INT) - 总飞行次数
- monetary (INT) - 总飞行里程
- r_score (INT, 1~5) - 最近消费评分
- f_score (INT, 1~5) - 消费频率评分
- m_score (INT, 1~5) - 消费金额评分
- cluster (INT) - 聚类标签（0,1,2,3）
- rfm_total (INT) - RFM总分
- avg_discount (DECIMAL) - 平均折扣率

任务：将用户的中文问题转换为合法的SELECT查询语句。
要求：
- 只返回SQL语句，不要解释
- 不要使用 ```sql 或 ``` 标记
- 只使用SELECT语句，禁止其他操作
- 字段名使用反引号包裹
- 中文字段筛选条件要正确处理
"""


def natural_to_sql(question: str, verbose: bool = True) -> str:
    """将自然语言问题转换为 SQL 查询语句"""
    full_prompt = f"{build_system_prompt()}\n问题：{question}\nSQL："
    sql = ask_gemma(full_prompt)

    # 清理 Markdown 标记
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "").strip()
    # 只取第一行，去掉多余注释
    sql = sql.split("\n")[0].strip() if "\n" in sql else sql.strip()

    # 安全校验：只允许 SELECT
    if not sql.lower().lstrip().startswith("select"):
        raise ValueError(f"生成的SQL不安全或格式错误（仅支持SELECT）：{sql[:100]}")

    if verbose:
        print(f"  自然语言: {question}")
        print(f"  生成SQL: {sql}")
    return sql


def execute_sql(sql: str) -> pd.DataFrame:
    """执行 SQL 查询，返回 DataFrame"""
    # 二次安全校验
    sql_stripped = sql.lstrip().lower()
    if not sql_stripped.startswith("select"):
        raise ValueError("仅允许执行 SELECT 查询")
    # 禁止危险关键词
    dangerous = ["delete", "drop", "truncate", "insert", "update", "alter", "create", "grant"]
    for keyword in dangerous:
        if re.search(rf"\b{keyword}\b", sql_stripped):
            raise ValueError(f"禁止执行包含 '{keyword}' 的语句")

    engine = create_engine(config.DATABASE_URL, echo=False)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    engine.dispose()
    return df


def query(question: str, verbose: bool = True) -> pd.DataFrame:
    """完整流程：自然语言 -> SQL -> 查询 -> 返回 DataFrame"""
    sql = natural_to_sql(question, verbose=verbose)
    df = execute_sql(sql)
    if verbose:
        print(f"  返回行数: {len(df)}")
        if not df.empty:
            print(f"  结果预览:")
            print(df.to_string(index=False))
    return df


def summarize_result(question: str, df: pd.DataFrame) -> str:
    """让模型用自然语言总结查询结果"""
    context = f"问题：{question}\n查询结果（前20行）：\n{df.head(20).to_string(index=False)}"
    prompt = f"根据以下数据和问题，用中文给出简洁的数据解读（2-3句话）：\n\n{context}\n\n解读："
    summary = ask_gemma(prompt)
    return summary


def interactive_mode():
    """交互式查询模式"""
    print("=" * 50)
    print("智能客户分析查询 (输入 'exit' 退出)")
    print("=" * 50)
    print()

    while True:
        try:
            question = input("\n请输入问题: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("再见！")
                break
            if not question:
                continue

            df = query(question)
            if not df.empty:
                print("\n数据解读:")
                summary = summarize_result(question, df)
                print(f"  {summary}")
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"  [错误] {e}")


if __name__ == "__main__":
    test_questions = [
        "统计各簇的平均年龄和平均总里程",
        "找出最近30天内未飞行的会员",
        "统计各性别会员数量",
    ]
    print("测试自然语言查询...")
    for q in test_questions:
        print(f"\n" + "-" * 40)
        try:
            query(q)
        except Exception as e:
            print(f"  [跳过] {e}")
    print("\n" + "=" * 40)
    print("若要进入交互模式，请运行:")
    print("  python llm_query.py --interactive")
