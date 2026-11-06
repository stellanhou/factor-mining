"""Official MCP Python SDK entry point for role-bound native chat tools."""

from __future__ import annotations

import argparse
from typing import Any

from .contracts import dumps, require
from .model import RecoverableToolError
from .native_chats import NativeChatAdapter, NativeRole, load_native_adapter

def create_server(adapter: NativeChatAdapter):
    """Expose the role's existing domain schemas through the official MCP SDK."""
    from mcp.server import Server, ServerRequestContext
    from mcp.types import (
        CallToolRequestParams,
        CallToolResult,
        ListToolsResult,
        PaginatedRequestParams,
        TextContent,
        Tool,
    )

    tools = [
        Tool(
            name=definition["name"],
            description=definition.get("description", ""),
            input_schema=definition["inputSchema"],
        )
        for definition in adapter.tool_definitions
    ]
    tool_names = {tool.name for tool in tools}

    async def list_tools(
        _context: ServerRequestContext,
        _params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        return ListToolsResult(tools=tools)

    async def call_tool(
        _context: ServerRequestContext,
        params: CallToolRequestParams,
    ) -> CallToolResult:
        if params.name not in tool_names:
            raise ValueError(f"tool is not available to fixed role {adapter.role}: {params.name}")
        arguments = params.arguments or {}
        try:
            result = adapter.call(params.name, arguments)
        except (RecoverableToolError, ValueError) as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                is_error=True,
            )
        serialized = dumps(result)
        return CallToolResult(
            content=[TextContent(type="text", text=serialized)],
            structured_content=result,
        )

    role_name = adapter.role.lower()
    return Server(
        name=f"factor-research-{role_name}",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve_stdio(adapter: NativeChatAdapter) -> None:
    from mcp.server.stdio import stdio_server

    server = create_server(adapter)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fixed-role factor research MCP server")
    parser.add_argument("--role", choices=("coordinator", "A", "B"), required=True)
    parser.add_argument("--binding", required=True, help="Trusted native chat binding JSON")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check-goal-evidence", action="store_true",
                           help=argparse.SUPPRESS)
    operation.add_argument("--action-control",
                           choices=("status", "begin-stop", "finish-stop", "resume"),
                           help=argparse.SUPPRESS)
    parser.add_argument("--action-id", help=argparse.SUPPRESS)
    parser.add_argument("--reason", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    adapter = load_native_adapter(args.binding, args.role)
    if args.check_goal_evidence:
        require(args.role == "coordinator", "Goal evidence checks are coordinator-only")
        result = {
            "research_id": adapter.binding.research_id,
            "run_root": str(adapter.binding.run_root),
            "contract_sha256": adapter.binding.contract_sha256,
            "goal": adapter.binding.goal,
            "evidence": adapter.actions.check_goal_evidence(),
        }
        print(dumps(result))
        return
    if args.action_control is not None:
        require(args.role == "coordinator", "run action control is coordinator-only")
        if args.action_control == "status":
            require(args.action_id is None and args.reason is None,
                    "action-control status does not accept action parameters")
            result = adapter.actions.ledger.status()
        elif args.action_control == "begin-stop":
            require(isinstance(args.action_id, str) and isinstance(args.reason, str),
                    "begin-stop requires an action ID and reason")
            result = adapter.actions.ledger.begin_stop(args.action_id, reason=args.reason)
        elif args.action_control == "finish-stop":
            require(isinstance(args.action_id, str) and isinstance(args.reason, str),
                    "finish-stop requires an action ID and reason")
            result = adapter.actions.ledger.finish_stop(args.action_id, reason=args.reason)
        else:
            require(isinstance(args.action_id, str) and args.reason is None,
                    "resume requires an action ID and no reason")
            result = adapter.actions.resume_actions(action_id=args.action_id)
        print(dumps(result))
        return

    import anyio

    anyio.run(serve_stdio, adapter)


if __name__ == "__main__":
    main()
