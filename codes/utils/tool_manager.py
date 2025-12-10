"""
LLM Tool Management Module

This module implements a `ToolManager` class designed to facilitate the integration 
of Python functions with Large Language Models (LLMs) that support function calling.
"""


import json
import inspect
from functools import wraps
from docstring_parser import parse
from rich import print
from typing import Callable, Any, Dict, get_origin, get_args, Union

from utils.tools import Tools


class ToolManager:
    def __init__(self, test=False) -> None:
        """
        Initializes the ToolManager.

        Args:
            test (bool): If True, enables debug printing during tool registration.
        """
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Dict] = {}
        self._test = test
        
    def _type_to_json_schema(self, type_hint: Any) -> dict:
        """
        Recursively converts Python type hints to JSON Schema format.
        
        Args:
            type_hint (Any): The Python type annotation to convert.
            
        Returns:
            dict: A dictionary representing the JSON Schema type.
        """
        if type_hint == int or type_hint == float:
            return {"type": "number"}
        if type_hint == str:
            return {"type": "string"}
        if type_hint == bool:
            return {"type": "boolean"}
        if type_hint is None or type_hint == type(None):
            return {"type": "null"}
        
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle list, e.g., list[str]
        if origin == list:
            if args:
                return {"type": "array", "items": self._type_to_json_schema(args[0])}
            return {"type": "array"} # Fallback for just 'list'

        # Handle tuple, e.g., tuple[str, int]
        if origin == tuple:
            if args:
                return {
                    "type": "array",
                    "items": {"type": "string"},  # Simplified to object items or string for now
                    "minItems": len(args),
                    "maxItems": len(args)
                }
            return {"type": "array", "items": {"type": "string"}}

        # Handle dict, e.g., dict[str, int]
        if origin == dict:
            if args and len(args) == 2:
                # JSON object keys must be strings. We only model the value type.
                return {"type": "object", "additionalProperties": self._type_to_json_schema(args[1])}
            return {"type": "object"}

        # Handle Union, e.g., Union[str, int]
        if origin == Union:
            # OpenAPI 3.0 uses oneOf, but simpler LLM tools might prefer a string description
            # For simplicity, we can describe it or default to the first type.
            return {
                "type": "object",
                "description": f"Can be one of types: {', '.join(str(a) for a in args)}"
            }

        # Fallback for complex or unsupported types
        return {"type": "string", "description": f"Type: {str(type_hint)}"}
    
    def register_tool(self, func: Callable) -> Callable:
        """
        Decorator to register a callable tool and build its JSON schema.
        
        It parses the function's docstring and type hints to generate a 
        description compatible with LLM function calling capabilities.

        Args:
            func (Callable): The function to register.

        Returns:
            Callable: The wrapped function.
        """
        tool_name = func.__name__
        
        # --- Generate Schema ---
        docstring = inspect.getdoc(func)
        if self._test:
            print(f"--- docstring ---\n\n{docstring}\n\n")
            
        parsed_docstring = parse(docstring)
        short_desc = parsed_docstring.short_description or ""
        long_desc = parsed_docstring.long_description or ""
        tool_description = f"{short_desc}\n{long_desc}".strip()
        
        # Map parameter names to their parsed docstring objects
        param_docs = {p.arg_name: p for p in parsed_docstring.params}
        if self._test:
            print(f"--- param_docs ---\n\n{param_docs}\n\n")
        
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            param_schema = self._type_to_json_schema(param.annotation)

            # Extract type name and description from docstring if available
            type_str = param_docs.get(name).type_name
            base_description = param_docs.get(name).description
            
            param_schema["description"] = f"({type_str}) {base_description}".strip()
            parameters["properties"][name] = param_schema
            
            # If no default value is provided, mark as required
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(name)
                
        self._tool_schemas[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters
        }
        if self._test:
            print(f"--- tool_schema ---\n\n{json.dumps(self._tool_schemas[tool_name], indent=4, ensure_ascii=False)}\n\n")
        
        # --- Register Tool Implementation ---
        self._tools[tool_name] = func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    def get_tools_json(self) -> str: 
        """
        Generates and returns a JSON string complying with LLM tool invocation specifications.
        
        This format is typically compatible with OpenAI's 'tools' parameter 
        (wrapped in {"type": "function", "function": ...}).

        Returns:
            str: JSON string of registered tools.
        """
        tools_json = []
        for schema in self._tool_schemas.values():
            tools_json.append({
                "type": "function",
                "function": schema
            })
        return json.dumps(tools_json, indent=4, ensure_ascii=False)
    
    def get_tools_json_responses_api(self) -> str: 
        """
        Generates and returns a JSON string of tool definitions in a flattened format.
        
        Unlike `get_tools_json`, this forces the top-level 'type' to be 'function'
        directly within the schema object.

        Returns:
            str: JSON string of tool schemas.
        """
        tools_json = []
        for schema in self._tool_schemas.values():
            schema_copy = schema.copy()
            schema_copy["type"] = "function"
            tools_json.append(schema_copy)
        return json.dumps(tools_json, indent=4, ensure_ascii=False)
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Calls a registered tool based on the LLM's request.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (Dict[str, Any]): A dictionary of arguments to pass to the tool.

        Returns:
            Any: The result of the tool execution, or an error message string if failed.
        """
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found. Avaliable tools: {list(self._tools.keys())}"
        
        tool_func = self._tools[tool_name]
        
        # try:
        result = tool_func(**arguments)
        return result
        # except TypeError as e:
        #     return f"Error: Invalid arguments for tool '{tool_name}'. {e}"
        # except Exception as e:
        #     return f"Error: An unexpected error occurred while executing tool '{tool_name}': {e}"
        
            
if __name__ == "__main__":
    
    manager = ToolManager()
    # Ensure utils.tools exists and has the Tools class
    tools_instance = Tools()
    
    manager.register_tool(tools_instance.search_wikipedia)