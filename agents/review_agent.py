"""
Code Review Agent - Reviews code and enforces quality standards.
"""
from typing import Dict, Tuple
from autogen import ConversableAgent
from utils.config import Config
from utils.logger import get_logger, log_agent_activity, log_api_call

logger = get_logger(__name__)


class CodeReviewAgent:
    """Agent responsible for reviewing code and providing feedback."""
    
    def __init__(self):
        """Initialize the Code Review Agent."""
        self.agent = ConversableAgent(
            name="code_reviewer",
            system_message="""You are a Senior Code Reviewer with expertise in Python, software engineering best practices, security, and code quality.

Your responsibilities:
1. Review Python code for correctness, efficiency, security, and edge cases
2. Check adherence to Python best practices (PEP 8, type hints, docstrings)
3. Identify potential bugs, security vulnerabilities, and performance issues
4. Ensure code handles edge cases appropriately
5. Verify code completeness (no placeholders, TODOs, or incomplete logic)

Review Criteria:
- Correctness: Does the code implement requirements correctly?
- Efficiency: Are there performance issues or inefficiencies?
- Security: Are there security vulnerabilities?
- Edge Cases: Are edge cases handled?
- Code Quality: Is the code clean, readable, and maintainable?
- Completeness: Is the code complete with no placeholders?

Approval Guidelines:
- APPROVE code if it correctly implements the core requirements, even if minor improvements are possible
- APPROVE code if it's functional, complete, and follows best practices reasonably well
- Only request changes for significant issues, bugs, or missing critical functionality
- Be practical: code doesn't need to be perfect, just production-ready

Output Format:
If code is APPROVED, respond with: "APPROVED" (at the start of your response)
If code needs changes, provide specific, actionable feedback in a clear format.""",
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
    
    def review(self, code: str, requirements: Dict) -> Tuple[bool, str]:
        """
        Review code and determine if it passes or needs revision.
        
        Args:
            code: Python code to review
            requirements: Original requirements dictionary
            
        Returns:
            Tuple of (is_approved: bool, feedback: str)
        """
        req_text = self._format_requirements(requirements)
        
        log_agent_activity(
            logger,
            "CodeReviewAgent",
            "Reviewing code",
            {"code_length": len(code), "requirements_count": len(requirements.get("functional_requirements", []))}
        )
        
        prompt = f"""Review the following Python code against these requirements:

REQUIREMENTS:
{req_text}

CODE TO REVIEW:
```python
{code}
```

Perform a thorough review. If the code is correct, complete, secure, and handles edge cases, respond with "APPROVED".
Otherwise, provide specific, actionable feedback on what needs to be fixed."""
        
        log_api_call(logger, "CodeReviewAgent", Config.MODEL, len(prompt))
        
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.agent.generate_reply(
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response is None:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Agent returned None response after retries.")
                
                feedback = response.get("content", "") if isinstance(response, dict) else str(response)
                break  # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise ValueError(f"Review API call failed after {max_retries} attempts: {str(e)}")
        feedback = feedback.strip()
        
        is_approved = feedback.upper().startswith("APPROVED")
        
        return is_approved, feedback
    
    def _format_requirements(self, requirements: Dict) -> str:
        """Format requirements for review context."""
        text = "FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("functional_requirements", []):
            text += f"- {req}\n"
        
        text += "\nNON-FUNCTIONAL REQUIREMENTS:\n"
        for req in requirements.get("non_functional_requirements", []):
            text += f"- {req}\n"
        
        return text

