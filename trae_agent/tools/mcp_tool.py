import mcp
from typing_extensions import override

from .base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class MCPTool(Tool):
    def __init__(self, client, tool: mcp.types.Tool, model_provider: str | None = None):
        super().__init__(model_provider)
        self.client = client
        self.tool = tool

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return self.tool.name

    @override
    def get_description(self) -> str:
        return self.tool.description

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        def properties_to_parameter():
            parameters = []
            inputSchema = self.tool.inputSchema
            required = inputSchema.get("required", [])
            properties = inputSchema.get("properties", {})
            for name, prop in properties.items():
                tool_para = ToolParameter(
                    name=name,
                    type=prop.get("type", "string"),
                    items=prop.get("items", None),
                    description=prop.get("description", ""),
                    required=name in required,
                )
                parameters.append(tool_para)
            return parameters

        return properties_to_parameter()

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            output = await self.client.call_tool(self.get_name(), arguments)
            # #9 修复:content 可能为图片 / EmbeddedResource / 空列表,
            # 直接 content[0].text 会 IndexError / AttributeError。
            # 安全提取所有文本块,无文本时给兜底。
            texts: list[str] = []
            for item in getattr(output, "content", None) or []:
                item_text = getattr(item, "text", None)
                if item_text:
                    texts.append(item_text)
            combined = "\n".join(texts)
            if output.isError:
                return ToolExecResult(
                    output=None,
                    error=combined or "MCP tool returned an error with no text content.",
                )
            else:
                return ToolExecResult(output=combined or "(no text content returned)")

        except Exception as e:
            return ToolExecResult(error=f"Error running mcp tool: {e}", error_code=-1)
