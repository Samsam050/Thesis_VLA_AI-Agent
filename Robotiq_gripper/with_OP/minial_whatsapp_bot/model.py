import os
from openai import OpenAI
from logger import setup_logger
import logging
from tools import make_schema, run_tool
import json

class OpenRouterModel():
    # return a model given the API (openrouter)
    def __init__(self, log_level=logging.INFO):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = "qwen/qwen3.5-plus-02-15" # currently, we choose cheap one to test.
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.logger = setup_logger('OpenRouterModel', log_level)
        self.tools = make_schema()  # Register tools from tools.py
        self.system_prompt = self._load_claude_md()

    def _load_claude_md(self):
        claude_md = os.path.join(os.path.dirname(__file__), "CLAUDE.md")
        if os.path.exists(claude_md):
            with open(claude_md) as f:
                content = f.read().strip()
            self.logger and self.logger.debug(f"Loaded CLAUDE.md as system prompt ({len(content)} chars)")
            return content
        return None

    def generate_response(self, messages):
        # Prepend system prompt from CLAUDE.md if available
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        self.logger.debug(f"Messages sent to model ({len(messages)} total):")
        for i, msg in enumerate(messages):
            content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
            self.logger.debug(f"  [{i}] {msg['role']}: {content[:100]}...")
        self.logger.debug("")

        # Agentic loop: keep calling tools until model returns text
        max_iterations = 20  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"Iteration {iteration}/{max_iterations}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                tools=self.tools,
                messages=messages,
                max_tokens=8192,
                temperature=0.7,
            )
            message = response.choices[0].message

            # Check if model wants to use tools
            if message.tool_calls:
                self.logger.info(f"Model requested {len(message.tool_calls)} tool call(s)")

                # Add assistant's message with tool calls to conversation
                messages.append(message.model_dump())

                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    tool_result = run_tool(tool_name, tool_args)
                    self.logger.debug(f"Tool result: {tool_result[:200]}...")

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Continue loop to get next response
                continue

            # No tool calls, return text response
            return message.content.strip() if message.content else ""

        # Max iterations reached
        self.logger.warning(f"Reached max iterations ({max_iterations})")
        return "Error: Maximum tool call iterations reached"