import os
from openai import OpenAI
from logger import setup_logger
import logging
from tools import make_schema, run_tool
import json
from dotenv import load_dotenv

load_dotenv()


class OpenAIModel:
    def __init__(self, log_level=logging.INFO):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.model_name = "gpt-5-nano"
        self.client = OpenAI(api_key=self.api_key)
        self.logger = setup_logger("OpenAIModel", log_level)
        self.tools = make_schema()
        self.system_prompt = self._load_claude_md()

    def _load_claude_md(self):
        claude_md = os.path.join(os.path.dirname(__file__), "CLAUDE.md")
        if os.path.exists(claude_md):
            with open(claude_md, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self.logger.debug(f"Loaded CLAUDE.md as system prompt ({len(content)} chars)")
            return content
        return None

    def generate_response(self, messages):
        input_items = []

        if self.system_prompt:
            input_items.append({"role": "system", "content": self.system_prompt})

        input_items.extend(messages)

        max_iterations = 20
        iteration = 0
        previous_response_id = None

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"Iteration {iteration}/{max_iterations}")

            kwargs = {
                "model": self.model_name,
                "tools": self.tools,
                "input": input_items,
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id

            response = self.client.responses.create(**kwargs)
            previous_response_id = response.id

            function_calls = [item for item in response.output if item.type == "function_call"]

            if function_calls:
                self.logger.info(f"Model requested {len(function_calls)} tool call(s)")
                tool_outputs = []

                for call in function_calls:
                    tool_name = call.name
                    tool_args = json.loads(call.arguments)

                    self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    tool_result = run_tool(tool_name, tool_args)
                    tool_result_str = tool_result if isinstance(tool_result, str) else json.dumps(tool_result)

                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": tool_result_str,
                        }
                    )

                input_items = tool_outputs
                continue

            return response.output_text or ""

        self.logger.warning(f"Reached max iterations ({max_iterations})")
        return "Error: Maximum tool call iterations reached"