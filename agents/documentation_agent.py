"""
Documentation Agent - Generates comprehensive Markdown documentation.
"""
from typing import Dict
from autogen import ConversableAgent
from utils.config import Config
from utils.logger import get_logger, log_agent_activity, log_api_call

logger = get_logger(__name__)


class DocumentationAgent:
    """Agent responsible for generating project documentation."""
    
    def __init__(self):
        """Initialize the Documentation Agent."""
        self.agent = ConversableAgent(
            name="documentation_writer",
            system_message="""You are a technical documentation specialist with expertise in software documentation.

Your task is to generate comprehensive, clear, and well-structured Markdown documentation for software projects.

Documentation MUST include these sections in order:

1. **Overview**: High-level description of what the code does, its purpose, and main features

2. **Agent Overview** (if this is a multi-agent system):
   - List all implemented agents
   - Describe each agent's responsibility
   - Explain how agents interact

3. **Workflow and Architecture**:
   - System architecture diagram (in text/ASCII or description)
   - Data flow between components
   - Execution workflow
   - How components interact

4. **Setup and Installation**:
   - Prerequisites
   - Step-by-step installation instructions
   - Environment setup
   - Configuration requirements

5. **How to Run the System**:
   - Command-line instructions
   - Required parameters
   - Example usage
   - Expected output

6. **Module Breakdown**: Detailed explanation of each module/class

7. **Function/Class Definitions**: Complete documentation with:
   - Parameters and their types
   - Return values and types
   - Exceptions that may be raised
   - Usage examples

8. **Usage Examples**: Practical, runnable examples

9. **Configuration**: Configuration options and environment variables

Write in clear, professional language suitable for both technical and non-technical audiences.
Use proper Markdown formatting with headers (##, ###), code blocks, lists, and tables where appropriate.""",
            llm_config={
                "config_list": [{
                    "model": Config.MODEL,
                    "api_key": Config.OPENAI_API_KEY,
                    "temperature": Config.TEMPERATURE,
                }],
                "timeout": 120,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
    
    def generate_documentation(self, code: str, requirements: Dict) -> str:
        """
        Generate comprehensive documentation for the code.
        
        Args:
            code: Python code to document
            requirements: Original requirements dictionary
            
        Returns:
            Markdown documentation string
        """
        req_text = self._format_requirements(requirements)
        
        log_agent_activity(logger, "DocumentationAgent", "Generating documentation", {"code_length": len(code)})
        
        prompt = f"""Generate comprehensive Markdown documentation for the following Python code.

ORIGINAL REQUIREMENTS:
{req_text}

GENERATED CODE:
```python
{code}
```

IMPORTANT: The documentation MUST include these sections in this order:

1. **Overview** - What the code does and its purpose
2. **Agent Overview** (if applicable) - List all agents and their responsibilities
3. **Workflow and Architecture** - System architecture, data flow, execution workflow
4. **Setup and Installation** - Prerequisites, installation steps, environment setup
5. **How to Run the System** - Command-line instructions, parameters, example usage
6. **Module Breakdown** - Detailed explanation of each module/class
7. **Function/Class Definitions** - Complete API documentation with parameters, return values, exceptions
8. **Usage Examples** - Practical, runnable code examples
9. **Configuration** - Configuration options and settings

Make the documentation structured, comprehensive, and production-ready. Use proper Markdown formatting."""
        
        log_api_call(logger, "DocumentationAgent", Config.MODEL, len(prompt))
        
        response = self.agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        
        if response is None:
            raise ValueError("Agent returned None response. Check API key and model configuration.")
        
        documentation = response.get("content", "") if isinstance(response, dict) else str(response)
        
        return documentation
    
    def _format_requirements(self, requirements: Dict) -> str:
        """Format requirements for documentation context."""
        text = "FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("functional_requirements", []):
            text += f"- {req}\n"
        
        text += "\nNON-FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("non_functional_requirements", []):
            text += f"- {req}\n"
        
        return text

