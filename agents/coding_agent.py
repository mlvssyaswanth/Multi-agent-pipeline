"""
Coding Agent - Generates clean, modular Python code from requirements.
"""
from typing import Dict, Any
from autogen import ConversableAgent
from utils.config import Config
from utils.logger import get_logger, log_agent_activity, log_api_call

logger = get_logger(__name__)


class CodingAgent:
    """Agent responsible for generating Python code from structured requirements."""
    
    def __init__(self):
        """Initialize the Coding Agent."""
        self.agent = ConversableAgent(
            name="coder",
            system_message="""You are an expert Python software engineer specializing in clean, modular, production-ready code.

PRIMARY MISSION:
Convert refined/structured requirements into clean, modular, functional Python code that works correctly.

CORE RESPONSIBILITIES:
1. **Convert Refined Requirements**: Transform structured requirements into working Python code
2. **Clean Code**: Write clean, readable, well-organized code
3. **Modular Design**: Create modular code with proper separation of concerns
4. **Functional Code**: Ensure code is functional, executable, and works as intended
5. **Python Best Practices**: Strictly follow Python best practices

PYTHON BEST PRACTICES (MANDATORY):
- **PEP 8 Compliance**: Follow PEP 8 style guide (naming conventions, line length, spacing)
- **Type Hints**: Include type hints for function parameters and return values
- **Docstrings**: Add comprehensive docstrings (Google or NumPy style) to all functions and classes
- **Code Organization**: Use proper module structure, imports, and organization
- **Error Handling**: Implement proper exception handling with try/except blocks
- **Input Validation**: Validate all inputs and handle edge cases
- **Naming Conventions**: Use descriptive, Pythonic names (snake_case for functions/variables, PascalCase for classes)
- **Code Comments**: Add inline comments where logic is complex
- **DRY Principle**: Don't Repeat Yourself - avoid code duplication
- **Single Responsibility**: Each function/class should have a single, clear purpose

CODE QUALITY REQUIREMENTS:
- **CLEAN**: Readable, well-formatted, properly indented code
- **MODULAR**: Separated into logical modules/functions/classes with clear responsibilities
- **FUNCTIONAL**: Code must work correctly and handle all specified requirements
- **EFFICIENCY**: Use appropriate algorithms and data structures
- **MAINTAINABILITY**: Well-documented, easy to understand and modify
- **ROBUSTNESS**: Handles errors gracefully with meaningful error messages

CRITICAL RULES:
- **DO NOT SELF-REVIEW**: You are ONLY responsible for code generation, NOT code review
- **NO REVIEW COMMENTS**: Do not add review comments, suggestions, or critiques
- **FOCUS ON IMPLEMENTATION**: Focus solely on writing working code
- **NO PLACEHOLDERS**: All code must be complete and functional, no TODOs or placeholders
- **NO INCOMPLETE CODE**: Ensure all functions are fully implemented
- **EXECUTABLE CODE**: Code must be ready to run without modification

MULTIPLE FILES:
If the requirements involve multiple files (e.g., main.py, utils.py, config.py), format them clearly:
- Start each file with: "# File: filename.py" on its own line
- Then provide the complete code for that file
- Separate files with a blank line and the next file header
- Example:
  # File: main.py
  [code for main.py]
  
  # File: utils.py
  [code for utils.py]

Output only the Python code, properly formatted and ready for execution.""",
            llm_config={
                "config_list": [{
                    "model": Config.MODEL,
                    "api_key": Config.OPENAI_API_KEY,
                    "temperature": Config.TEMPERATURE,
                }],
                "timeout": 180,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
    
    def generate_code(self, requirements: Dict[str, Any], feedback: str = None, previous_code: str = None) -> str:
        """
        Generate Python code from structured requirements.
        
        Args:
            requirements: Structured requirements dictionary
            feedback: Optional feedback from code review agent
            previous_code: Optional previous code for follow-up prompts (to modify instead of generating from scratch)
            
        Returns:
            Generated Python code as string
        """
        req_text = self._format_requirements(requirements)
        
        log_agent_activity(
            logger, 
            "CodingAgent", 
            "Generating code",
            {"has_feedback": bool(feedback), "requirements_count": len(requirements.get("functional_requirements", []))}
        )
        
        # Build prompt based on whether we have feedback, previous code, or neither
        if feedback:
            prompt = f"""Convert the following refined requirements into clean, modular, functional Python code.

REQUIREMENTS:
{req_text}

REVIEW FEEDBACK (address these issues):
{feedback}

TASK:
Generate improved Python code that:
1. Converts the refined requirements into working code
2. Addresses all issues mentioned in the review feedback
3. Follows Python best practices (PEP 8, type hints, docstrings)
4. Is clean, modular, and functional
5. Is complete and executable

CRITICAL: 
- DO NOT review or critique the code - only implement it
- Focus on code generation, not code review
- Make the code work correctly based on requirements and feedback"""
        elif previous_code:
            prompt = f"""Modify the following existing code based on the updated requirements.

PREVIOUS CODE:
{previous_code}

UPDATED REQUIREMENTS:
{req_text}

TASK:
Modify the existing code to:
1. Incorporate the updated requirements while maintaining existing functionality
2. Make necessary changes, additions, or modifications as specified
3. Keep the code structure and style consistent with the previous code
4. Follow Python best practices (PEP 8, type hints, docstrings)
5. Ensure the code is complete and executable

CRITICAL: 
- Modify the existing code rather than rewriting from scratch
- Maintain consistency with the previous code structure
- Only change what is necessary based on the updated requirements
- Keep all working functionality that isn't being modified"""
        else:
            prompt = f"""Convert the following refined requirements into clean, modular, functional Python code.

REQUIREMENTS:
{req_text}

TASK:
Generate Python code that:
1. Converts the refined requirements into working code
2. Follows Python best practices (PEP 8, type hints, docstrings, proper error handling)
3. Is clean, modular, and well-organized
4. Is functional and executable
5. Is complete with no placeholders or TODOs

PYTHON BEST PRACTICES TO FOLLOW:
- PEP 8 style guide compliance
- Type hints for all functions
- Comprehensive docstrings
- Proper error handling
- Input validation
- Modular design with separation of concerns
- Meaningful variable and function names

CRITICAL: 
- DO NOT review or critique the code - only implement it
- Focus on code generation, not code review
- Generate working, functional code"""
        
        log_api_call(logger, "CodingAgent", Config.MODEL, len(prompt))
        
        import time
        max_retries = 3
        code = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.agent.generate_reply(
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response is None:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise ValueError("Agent returned None response after retries. This may be due to API rate limiting or model unavailability.")
                
                # Extract content from response - handle different response formats
                if isinstance(response, dict):
                    code = response.get("content", "") or response.get("text", "") or str(response)
                else:
                    code = str(response)
                
                # Log response length for debugging
                logger.debug(f"CodingAgent: Received response length: {len(code)} characters")
                
                if not code or not code.strip():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise ValueError("Agent returned empty code after retries.")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise ValueError(f"API call failed after {max_retries} attempts: {str(e)}. Check API key, model configuration, and network connection.")
        
        if not code:
            error_msg = f"Failed to generate code after {max_retries} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise ValueError(error_msg)
        
        # Extract code blocks from the response
        extracted_code = self._extract_code_blocks(code)
        
        # Log extraction results for debugging
        logger.debug(f"CodingAgent: Extracted code length: {len(extracted_code)} characters (original: {len(code)})")
        
        # Safety check: if extraction seems incomplete, try to use more of the original content
        if len(extracted_code) < len(code) * 0.3 and len(code) > 200:
            # If extracted code is very short compared to original, extraction might have failed
            logger.warning(f"CodingAgent: Extracted code ({len(extracted_code)} chars) is much shorter than original ({len(code)} chars). Using full content as fallback.")
            # Check if original content looks like code (has Python keywords)
            if any(keyword in code for keyword in ['def ', 'class ', 'import ', 'from ', '# File:']):
                # Use original content if it looks like code
                return code.strip()
        
        return extracted_code if extracted_code else code.strip()
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements dictionary into readable text."""
        text = "FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("functional_requirements", []):
            text += f"- {req}\n"
        
        text += "\nNON-FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("non_functional_requirements", []):
            text += f"- {req}\n"
        
        text += "\nASSUMPTIONS:\n"
        for assumption in requirements.get("assumptions", []):
            text += f"- {assumption}\n"
        
        text += "\nCONSTRAINTS:\n"
        for constraint in requirements.get("constraints", []):
            text += f"- {constraint}\n"
        
        return text
    
    def _extract_code_blocks(self, content: str) -> str:
        """
        Extract Python code from markdown code blocks.
        Handles multiple code blocks and incomplete responses.
        """
        if not content:
            return ""
        
        # Try to find all code blocks
        code_blocks = []
        
        # Pattern 1: Look for ```python blocks
        python_block_pattern = r'```python\s*\n(.*?)(?:```|$)'
        import re
        python_matches = re.finditer(python_block_pattern, content, re.DOTALL)
        for match in python_matches:
            code = match.group(1).strip()
            if code:
                code_blocks.append(code)
        
        # Pattern 2: If no python blocks, look for generic ``` blocks
        if not code_blocks:
            generic_block_pattern = r'```(?:[a-z]+)?\s*\n(.*?)(?:```|$)'
            generic_matches = re.finditer(generic_block_pattern, content, re.DOTALL)
            for match in generic_matches:
                code = match.group(1).strip()
                if code:
                    code_blocks.append(code)
        
        # Pattern 3: If still no blocks found, check if content looks like code
        if not code_blocks:
            # Check if content starts with common Python keywords or imports
            lines = content.strip().split('\n')
            if lines and (lines[0].startswith('import ') or 
                         lines[0].startswith('from ') or
                         lines[0].startswith('def ') or
                         lines[0].startswith('class ') or
                         lines[0].startswith('#') or
                         any(line.strip().startswith(('def ', 'class ', 'import ', 'from ')) for line in lines[:5])):
                # Likely code without markdown blocks - return as is
                return content.strip()
        
        # Combine all code blocks (for multiple files scenario)
        if code_blocks:
            # If multiple blocks, join them with file markers if they look like separate files
            combined = '\n\n'.join(code_blocks)
            return combined
        
        # Fallback: return original content if no code blocks found
        return content.strip()

