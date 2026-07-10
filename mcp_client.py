"""
MCP Client 模块
负责与 api.py (MCP Server) 通信，调用工具
"""
import requests
import json
from typing import Dict, Any, Optional


class MCPClient:
    """MCP 客户端，通过 HTTP 调用 MCP Server 的工具"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:5001"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """检查 MCP Server 是否可用"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_tools(self) -> list:
        """获取所有可用工具的定义"""
        try:
            response = self.session.get(f"{self.server_url}/tools/list", timeout=5)
            if response.status_code == 200:
                return response.json().get("tools", [])
            return []
        except Exception as e:
            print(f"[MCP Client] list_tools 失败: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用指定工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数字典
            
        Returns:
            工具执行结果
        """
        try:
            response = self.session.post(
                f"{self.server_url}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments
                },
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # 检查是否成功
                if not response_data.get("success"):
                    return {
                        "error": response_data.get("error", "工具调用失败"),
                        "tool": tool_name
                    }
                
                # api.py 已经解析了 MCP 格式，result 就是纯工具返回值
                return response_data.get("result", {})
            else:
                return {
                    "error": f"MCP Server 返回 {response.status_code}",
                    "detail": response.text
                }
        except Exception as e:
            return {
                "error": f"调用工具 {tool_name} 失败",
                "detail": str(e)
            }


# 全局 MCP Client 实例
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """获取全局 MCP Client 实例"""
    global _mcp_client
    if _mcp_client is None:
        server_url = getattr(__import__('config'), 'MCP_SERVER_URL', 'http://127.0.0.1:5001')
        _mcp_client = MCPClient(server_url)
    return _mcp_client


def is_mcp_available() -> bool:
    """检查 MCP Server 是否可用"""
    try:
        client = get_mcp_client()
        return client.health_check()
    except Exception:
        return False
