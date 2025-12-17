"""
Test Case Generation Agent - Generates executable pytest test cases.
"""
from typing import Dict
from autogen import ConversableAgent
from utils.config import Config
from utils.logger import get_logger, log_agent_activity, log_api_call

logger = get_logger(__name__)


class TestGenerationAgent:
    """Agent responsible for generating executable pytest test cases."""
    
    def __init__(self):
        """Initialize the Test Generation Agent."""
        system_msg = """You are a Senior Test Engineer specializing in Python testing with pytest.

PRIMARY MISSION:
Generate BOTH unit tests AND integration tests that are pytest-compatible, executable without modification, and designed to PASS with the generated code.

CORE RESPONSIBILITIES:
1. **Generate Unit Tests**: Create unit tests that test individual functions, methods, and classes in isolation
2. **Generate Integration Tests**: Create integration tests that test how multiple components work together
3. **At Least One Test Per Module**: MANDATORY - Create at least one test case per module/class/function
4. **pytest-Compatible**: Generate test files that are fully compatible with pytest framework
5. **Executable Without Modification**: Tests must be immediately runnable with pytest without any changes
6. **Tests Must Pass**: All tests must be designed to PASS with the generated code - use correct imports, correct function names, and realistic test data

MANDATORY REQUIREMENTS:

1. **UNIT TESTS** (REQUIRED):
   - Generate unit tests (test individual functions, methods, classes in isolation)
   - Test one unit of code at a time
   - Use mocks/fixtures where appropriate to isolate units
   - Focus on testing individual components, not full system integration
   - Mark with comment: "# Unit Tests" at the beginning of unit test section
   - Test each function/method/class independently
   - Use correct imports matching the actual code structure
   - Use realistic test data that will work with the actual implementation

2. **INTEGRATION TESTS** (REQUIRED):
   - Generate integration tests that test how multiple components work together
   - Test interactions between different modules, classes, or functions
   - Test end-to-end workflows and data flow
   - Test how different parts of the system integrate
   - Mark with comment: "# Integration Tests" at the beginning of integration test section
   - Use actual imports and real component interactions (not mocks for integration)
   - Test realistic scenarios that demonstrate the system working together

3. **AT LEAST ONE TEST PER MODULE**:
   - Identify ALL modules, classes, and functions in the code
   - Generate a MINIMUM of one test for each identified component
   - If code has multiple classes, test each class
   - If code has multiple functions, test each function
   - If code has multiple modules (separate files), ensure tests for each module
   - Group related tests by module/class using test classes or clear naming

3. **PYTEST-COMPATIBLE TEST FILES**:
   - Use proper pytest syntax (test functions starting with `test_`)
   - Use pytest assertions (`assert` statements)
   - Use pytest fixtures if needed (`@pytest.fixture`)
   - Use pytest parametrization if appropriate (`@pytest.mark.parametrize`)
   - Use pytest exception testing (`pytest.raises()`)
   - Follow pytest naming conventions (test files: `test_*.py`, test functions: `test_*`)
   - Import pytest properly: `import pytest`
   - Use pytest-compatible imports and structure

4. **EXECUTABLE WITHOUT MODIFICATION**:
   - All imports must be correct and available
   - No placeholders, TODOs, or incomplete tests
   - All test code must be syntactically correct
   - Tests must run with `pytest` command without any code changes
   - Import statements must match the actual code structure
   - Test data must be self-contained or properly mocked
   - No missing dependencies or undefined variables

TEST REQUIREMENTS:
- Use proper pytest syntax and assertions
- Include descriptive test names that indicate what module/function is being tested
- Test both success and failure scenarios
- Cover normal cases, edge cases, and error scenarios
- For each test, include a comment showing expected execution result
- Test actual code execution, not just imports
- Use pytest best practices (fixtures, parametrization where appropriate)

OUTPUT FORMAT:
- Python test code fully compatible with pytest
- MUST include BOTH unit tests AND integration tests
- Separate sections clearly with comments: "# Unit Tests" and "# Integration Tests"
- Include execution results as comments after each test
- Format: # Expected Result: [description of what should happen]
- Organize tests by module/class for clarity
- Ready to run with: `pytest test_file.py`
- All tests must be designed to PASS with the generated code

STRUCTURE:
```python
# Unit Tests
import pytest
# ... unit test code here ...

# Integration Tests
# ... integration test code here ...
```

Example:
```python
import pytest

# Unit tests for calculator module
def test_calculator_addition():
    \"\"\"Test addition function.\"\"\"
    result = add(2, 3)
    assert result == 5
    # Expected Result: Test passes, returns 5

def test_calculator_division():
    \"\"\"Test division function.\"\"\"
    result = divide(10, 2)
    assert result == 5
    # Expected Result: Test passes, returns 5

def test_calculator_division_by_zero():
    \"\"\"Test division by zero raises ValueError.\"\"\"
    with pytest.raises(ValueError):
        divide(10, 0)
    # Expected Result: Test passes, ValueError raised correctly
```

Output only the Python test code, properly formatted and ready for execution with pytest."""
        self.agent = ConversableAgent(
            name="test_generator",
            system_message=system_msg,
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
    
    def generate_tests(self, code: str, requirements: Dict) -> str:
        """
        Generate pytest test cases for the given code.
        
        Args:
            code: Python code to test
            requirements: Original requirements dictionary
            
        Returns:
            Generated pytest test code as string
        """
        req_text = self._format_requirements(requirements)
        
        log_agent_activity(logger, "TestGenerationAgent", "Generating test cases", {"code_length": len(code)})
        
        # Analyze code to identify modules/classes/functions
        modules_info = self._identify_modules(code)
        
        prompt = f"""Generate BOTH unit tests AND integration tests for the following Python code. Tests must be pytest-compatible, executable without modification, and designed to PASS with the generated code.

ORIGINAL REQUIREMENTS:
{req_text}

CODE TO TEST:
```python
{code}
```

IDENTIFIED MODULES/CLASSES/FUNCTIONS:
{modules_info}

MANDATORY REQUIREMENTS:

1. **GENERATE UNIT TESTS** (REQUIRED):
   - Create unit tests (test individual functions, methods, classes in isolation)
   - Test one unit of code at a time
   - Use mocks/fixtures where appropriate to isolate units
   - Focus on testing individual components
   - Mark the section with: "# Unit Tests" at the beginning
   - Use correct imports matching the actual code structure
   - Use realistic test data that will work with the actual implementation
   - Ensure all unit tests will PASS with the generated code

2. **GENERATE INTEGRATION TESTS** (REQUIRED):
   - Create integration tests that test how multiple components work together
   - Test interactions between different modules, classes, or functions
   - Test end-to-end workflows and data flow
   - Test how different parts of the system integrate
   - Mark the section with: "# Integration Tests" at the beginning
   - Use actual imports and real component interactions (not mocks for integration)
   - Test realistic scenarios that demonstrate the system working together
   - Ensure all integration tests will PASS with the generated code

3. **AT LEAST ONE TEST PER MODULE** (MANDATORY):
   - **You MUST create at least one test for each module/class/function identified above**
   - If multiple modules/classes/functions exist, ensure each has at least one test
   - Test all major functions and classes
   - Group related tests by module/class

4. **PYTEST-COMPATIBLE TEST FILES**:
   - Use proper pytest syntax (test functions starting with `test_`)
   - Use pytest assertions (`assert` statements)
   - Import pytest: `import pytest`
   - Use pytest.raises() for exception testing
   - Follow pytest naming conventions
   - Use pytest fixtures if needed
   - Ensure full pytest compatibility

5. **EXECUTABLE WITHOUT MODIFICATION**:
   - All imports must be correct and match the actual code structure
   - No placeholders, TODOs, or incomplete tests
   - All test code must be syntactically correct
   - Tests must run with `pytest` command without any code changes
   - Test data must be self-contained or properly mocked
   - No missing dependencies or undefined variables

ADDITIONAL REQUIREMENTS:
- Cover normal cases, edge cases, and error scenarios
- Include descriptive test names that indicate which module/function is being tested
- Include execution results as comments showing what each test should produce
- Test the actual execution of the code, not just imports
- Include both positive and negative test cases
- Make tests comprehensive and realistic

For each test function, add a comment showing the expected execution result, for example:
# Expected Result: Test passes, function returns correct value
# Expected Result: Test passes, exception is raised correctly

CRITICAL: 
- **You MUST create BOTH unit tests AND integration tests**
- **You MUST create at least one unit test for each module/class/function identified**
- **All tests must be designed to PASS with the generated code**
- Tests must be pytest-compatible and runnable with `pytest` without modification
- All imports and dependencies must be correct and match the actual code
- Tests must be complete and functional
- Use correct function names, class names, and module names from the actual code
- Use realistic test data that matches what the code expects

OUTPUT STRUCTURE:
```python
# Unit Tests
import pytest
# ... unit test code here ...

# Integration Tests
# ... integration test code here ...
```

Output only the Python test code, properly formatted, pytest-compatible, with both unit and integration tests, and ready for execution. All tests must pass with the generated code."""
        
        log_api_call(logger, "TestGenerationAgent", Config.MODEL, len(prompt))
        
        import time
        max_retries = 3
        test_code = None
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
                    test_code = response.get("content", "") or response.get("text", "") or str(response)
                else:
                    test_code = str(response)
                
                # Log response length for debugging
                logger.debug(f"TestGenerationAgent: Received response length: {len(test_code)} characters")
                
                if not test_code or not test_code.strip():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise ValueError("Agent returned empty test code after retries.")
                
                extracted_code = self._extract_code_blocks(test_code)
                
                # Log extraction results for debugging
                logger.debug(f"TestGenerationAgent: Extracted code length: {len(extracted_code)} characters (original: {len(test_code)})")
                
                # Safety check: if extraction seems incomplete, try to use more of the original content
                if len(extracted_code) < len(test_code) * 0.3 and len(test_code) > 200:
                    # If extracted code is very short compared to original, extraction might have failed
                    logger.warning(f"TestGenerationAgent: Extracted code ({len(extracted_code)} chars) is much shorter than original ({len(test_code)} chars). Using full content as fallback.")
                    # Check if original content looks like code (has Python keywords)
                    if any(keyword in test_code for keyword in ['def test_', 'import pytest', 'class Test', 'assert ']):
                        # Use original content if it looks like test code
                        test_code = test_code.strip()
                    else:
                        test_code = extracted_code
                else:
                    test_code = extracted_code
                
                # Final fallback
                if not test_code or not test_code.strip():
                    test_code = test_code if test_code else (extracted_code if extracted_code else test_code)
                
                if not test_code or not test_code.strip():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise ValueError("Extracted test code is empty after retries.")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise ValueError(f"Test generation API call failed after {max_retries} attempts: {str(e)}. Check API key, model configuration, and network connection.")
        
        if not test_code:
            error_msg = f"Failed to generate test cases after {max_retries} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise ValueError(error_msg)
        
        return test_code
    
    def _format_requirements(self, requirements: Dict) -> str:
        """Format requirements for test generation context."""
        text = "FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("functional_requirements", []):
            text += f"- {req}\n"
        
        text += "\nNON-FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("non_functional_requirements", []):
            text += f"- {req}\n"
        
        return text
    
    def _identify_modules(self, code: str) -> str:
        """
        Identify modules, classes, and functions in the code.
        
        Args:
            code: Python code string
            
        Returns:
            Formatted string listing identified modules/classes/functions
        """
        import re
        import ast
        
        modules_info = []
        
        try:
            # Try to parse the code as AST
            tree = ast.parse(code)
            
            # Find all classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            
            # Find all function definitions (not methods)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's a top-level function (not inside a class)
                    parent = None
                    for parent_node in ast.walk(tree):
                        if isinstance(parent_node, (ast.ClassDef, ast.FunctionDef)):
                            for child in ast.iter_child_nodes(parent_node):
                                if child == node:
                                    parent = parent_node
                                    break
                    if not isinstance(parent, ast.ClassDef):
                        functions.append(node.name)
            
            if classes:
                modules_info.append(f"Classes found: {', '.join(classes)}")
            if functions:
                modules_info.append(f"Top-level functions found: {', '.join(functions)}")
            
            # Also check for multiple files pattern
            if "# File:" in code or "## File:" in code:
                file_pattern = r'#+\s*File:\s*([^\n]+\.py)'
                files = re.findall(file_pattern, code)
                if files:
                    modules_info.append(f"Files found: {', '.join(files)}")
            
        except SyntaxError:
            # If AST parsing fails, use regex fallback
            # Find class definitions
            class_pattern = r'^class\s+(\w+)'
            classes = re.findall(class_pattern, code, re.MULTILINE)
            if classes:
                modules_info.append(f"Classes found: {', '.join(classes)}")
            
            # Find function definitions (not indented)
            func_pattern = r'^def\s+(\w+)\s*\('
            functions = re.findall(func_pattern, code, re.MULTILINE)
            if functions:
                modules_info.append(f"Top-level functions found: {', '.join(functions)}")
            
            # Check for multiple files
            if "# File:" in code or "## File:" in code:
                file_pattern = r'#+\s*File:\s*([^\n]+\.py)'
                files = re.findall(file_pattern, code)
                if files:
                    modules_info.append(f"Files found: {', '.join(files)}")
        
        if not modules_info:
            modules_info.append("Single module detected (no explicit classes or multiple files found)")
        
        return "\n".join(modules_info) if modules_info else "Code structure analysis completed"
    
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
        
        # Combine all code blocks (for multiple test files scenario)
        if code_blocks:
            # If multiple blocks, join them with file markers if they look like separate files
            combined = '\n\n'.join(code_blocks)
            return combined
        
        # Fallback: return original content if no code blocks found
        return content.strip()

