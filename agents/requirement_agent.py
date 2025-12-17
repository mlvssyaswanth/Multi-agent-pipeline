"""
Requirement Analysis Agent - Converts natural language to structured requirements.
Detects ambiguity and asks clarifying questions.
"""
import json
import re
from typing import Dict, Any, List, Optional
from autogen import ConversableAgent
from utils.config import Config
from utils.logger import get_logger, log_agent_activity, log_api_call

logger = get_logger(__name__)


class RequirementAnalysisAgent:
    """Agent responsible for analyzing and structuring user requirements."""
    
    def __init__(self):
        """Initialize the Requirement Analysis Agent."""
        self.agent = ConversableAgent(
            name="requirement_analyst",
            system_message="""You are a Senior Requirements Analyst specializing in software engineering.
Your task is to analyze natural language requirements and convert vague, ambiguous inputs into structured, actionable software requirements.

CRITICAL CAPABILITIES:
1. **Ambiguity Detection**: Identify vague terms, missing details, unclear specifications
2. **Clarifying Questions**: Generate specific questions to resolve ambiguity (even if simulated/answered automatically)
3. **Structured Output**: Convert requirements into clear, structured format

OUTPUT FORMAT:
You must output a JSON object with the following structure:
{
    "functional_requirements": ["list of specific, testable functional requirements"],
    "non_functional_requirements": ["list of non-functional requirements (performance, security, usability, etc.)"],
    "assumptions": ["list of assumptions made when requirements are vague"],
    "constraints": ["list of constraints identified (technical, business, time, etc.)"],
    "programming_language": "detected programming language (e.g., 'python', 'javascript', 'java', 'cpp', 'go', 'rust', etc.) or 'python' if not specified",
    "clarifying_questions": [
        {
            "question": "the clarifying question text",
            "assumption": "the assumption made to proceed without clarification",
            "code": "code snippet or example showing how this assumption is implemented"
        }
    ],
    "ambiguity_detected": true/false,
    "ambiguity_notes": "description of detected ambiguities and how they were resolved"
}

IMPORTANT FOR LANGUAGE DETECTION:
- Detect the programming language from the user input
- Look for explicit mentions: "in Python", "using JavaScript", "Java code", "C++", etc.
- Look for language-specific terms: "npm" (JavaScript), "pip" (Python), "package.json" (JavaScript), "pom.xml" (Java), etc.
- Look for file extensions mentioned: ".js", ".py", ".java", ".cpp", ".go", ".rs", etc.
- If no language is specified, default to "python"
- Common languages: python, javascript, typescript, java, cpp, csharp, go, rust, ruby, php, swift, kotlin

IMPORTANT FOR CLARIFYING QUESTIONS:
- Each clarifying question must be an object with "question", "assumption", and "code" fields
- The "assumption" field should explain what assumption was made to proceed
- The "code" field should contain a relevant code snippet or example showing how the assumption is implemented
- If no code is applicable, use a comment explaining the assumption instead

AMBIGUITY DETECTION:
Look for:
- Vague terms: "user-friendly", "fast", "good", "easy", "simple"
- Missing specifications: no input/output formats, no error handling mentioned, no UI details
- Unclear scope: "some features", "various operations", "multiple ways"
- Missing constraints: no performance requirements, no platform specified, no security mentioned
- Unclear user roles: who are the users? what permissions?

CLARIFYING QUESTIONS:
When ambiguity is detected, generate specific questions such as:
- "What specific input format should be accepted?"
- "What are the performance requirements (response time, throughput)?"
- "What platform should this run on?"
- "Who are the target users?"
- "What error handling is expected?"
- "What is the expected output format?"

Be thorough, specific, and ensure all requirements are testable and implementable.
Focus on clarity, completeness, and identifying all ambiguities.""",
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
    
    def analyze(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user input and return structured requirements.
        Detects ambiguity and generates clarifying questions.
        
        Args:
            user_input: Natural language description of requirements
            context: Optional context dictionary containing previous prompts and results for follow-up prompts
            
        Returns:
            Dictionary containing structured requirements with ambiguity detection
        """
        log_agent_activity(logger, "RequirementAnalysisAgent", "Starting analysis", {"input_length": len(user_input), "has_context": context is not None})
        
        # First, detect ambiguity
        ambiguity_info = self._detect_ambiguity(user_input)
        
        # Build context information if available
        context_section = ""
        if context and context.get("is_active"):
            previous_prompts = context.get("previous_prompts", [])
            previous_results = context.get("previous_results")
            
            if previous_prompts or previous_results:
                context_section = "\n\nPREVIOUS CONTEXT:\n"
                if previous_prompts:
                    context_section += f"Previous prompt(s): {previous_prompts[-1]}\n"
                if previous_results:
                    prev_reqs = previous_results.get("requirements", {})
                    if prev_reqs:
                        context_section += f"Previous functional requirements: {', '.join(prev_reqs.get('functional_requirements', [])[:3])}\n"
                    prev_code = previous_results.get("code", "")
                    if prev_code:
                        code_summary = prev_code[:200] + "..." if len(prev_code) > 200 else prev_code
                        context_section += f"Previous code summary: {code_summary}\n"
                context_section += "\nThis is a follow-up request. Please update/modify the requirements based on the new input while maintaining consistency with the previous context.\n"
        
        prompt = f"""Analyze the following user requirement and convert vague natural language into structured, actionable software requirements.
{context_section}
USER REQUIREMENT:
{user_input}

TASK:
1. **Detect Ambiguity**: Identify vague terms, missing details, unclear specifications
2. **Generate Clarifying Questions**: Create specific questions to resolve any ambiguity (even if simulated/answered automatically)
3. **Convert to Structured Requirements**: Transform the requirement into clear, testable requirements

OUTPUT FORMAT:
Provide your analysis as a JSON object with this exact structure:
{{
    "functional_requirements": ["specific, testable functional requirements"],
    "non_functional_requirements": ["non-functional requirements (performance, security, usability, scalability, etc.)"],
    "assumptions": ["assumptions made when requirements are vague or incomplete"],
    "constraints": ["constraints identified (technical, business, time, platform, etc.)"],
    "programming_language": "detected programming language (e.g., 'python', 'javascript', 'java', 'cpp', 'go', 'rust', etc.) or 'python' if not specified",
    "clarifying_questions": [
        {{
            "question": "the clarifying question text",
            "assumption": "the assumption made to proceed without clarification",
            "code": "code snippet or example showing how this assumption is implemented"
        }}
    ],
    "ambiguity_detected": true/false,
    "ambiguity_notes": "description of detected ambiguities and how assumptions were made to resolve them"
}}

IMPORTANT FOR LANGUAGE DETECTION:
- Detect the programming language from the user input
- Look for explicit mentions: "in Python", "using JavaScript", "Java code", "C++", etc.
- Look for language-specific terms: "npm" (JavaScript), "pip" (Python), "package.json" (JavaScript), "pom.xml" (Java), etc.
- Look for file extensions mentioned: ".js", ".py", ".java", ".cpp", ".go", ".rs", etc.
- If no language is specified, default to "python"
- Common languages: python, javascript, typescript, java, cpp, csharp, go, rust, ruby, php, swift, kotlin

IMPORTANT:
- If ambiguity is detected, generate clarifying questions AND make reasonable assumptions
- Each clarifying question MUST be an object with "question", "assumption", and "code" fields
- The "assumption" field should explain what assumption was made to proceed with this question
- The "code" field should contain a relevant code snippet, example, or comment showing how the assumption is implemented
- Document all assumptions clearly
- Ensure functional requirements are specific and testable
- Include non-functional requirements even if not explicitly mentioned (make reasonable assumptions)
- Be thorough and comprehensive"""
        
        log_api_call(logger, "RequirementAnalysisAgent", Config.MODEL, len(prompt))
        
        import time
        max_retries = 3
        content = None
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
                    logger.error("Agent returned None response after retries")
                    raise ValueError("Agent returned None response after retries. This may be due to API rate limiting or model unavailability.")
                
                content = response.get("content", "") if isinstance(response, dict) else str(response)
                
                if not content or not content.strip():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise ValueError("Agent returned empty content after retries.")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise ValueError(f"Requirement analysis API call failed after {max_retries} attempts: {str(e)}. Check API key, model configuration, and network connection.")
        
        if not content:
            error_msg = f"Failed to analyze requirements after {max_retries} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise ValueError(error_msg)
        
        log_api_call(logger, "RequirementAnalysisAgent", Config.MODEL, len(prompt), len(content))
        
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                requirements = json.loads(json_str)
            else:
                requirements = self._parse_fallback(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {str(e)}, using fallback parser")
            requirements = self._parse_fallback(content)
        
        # Ensure all required fields are present
        # Handle both old format (list of strings) and new format (list of objects)
        clarifying_questions_raw = requirements.get("clarifying_questions", [])
        clarifying_questions = []
        
        for q in clarifying_questions_raw:
            if isinstance(q, dict):
                # New format with question, assumption, code
                clarifying_questions.append({
                    "question": q.get("question", ""),
                    "assumption": q.get("assumption", ""),
                    "code": q.get("code", "")
                })
            elif isinstance(q, str):
                # Old format - convert to new format
                clarifying_questions.append({
                    "question": q,
                    "assumption": "Assumption not specified",
                    "code": "# No code example provided"
                })
        
        # Detect programming language from user input if not in requirements
        detected_language = requirements.get("programming_language", "")
        if not detected_language:
            detected_language = self._detect_programming_language(user_input)
        else:
            # Normalize the language from requirements
            detected_language = detected_language.lower()
        
        # Double-check: If user mentions React but language is JavaScript, upgrade to React
        user_lower = user_input.lower()
        if detected_language == "javascript" and any(term in user_lower for term in ["react", "jsx", "reactjs", "react.js"]):
            detected_language = "react"
            logger.info("Upgraded JavaScript to React based on user input")
        
        # Final language (use detected if requirements didn't have it)
        final_language = detected_language.lower() if detected_language else "python"
        
        result = {
            "functional_requirements": requirements.get("functional_requirements", []),
            "non_functional_requirements": requirements.get("non_functional_requirements", []),
            "assumptions": requirements.get("assumptions", []),
            "constraints": requirements.get("constraints", []),
            "programming_language": final_language,
            "clarifying_questions": clarifying_questions,
            "ambiguity_detected": requirements.get("ambiguity_detected", False),
            "ambiguity_notes": requirements.get("ambiguity_notes", ""),
        }
        
        if result["ambiguity_detected"]:
            logger.info(f"Ambiguity detected, {len(result['clarifying_questions'])} questions generated")
        
        logger.info(f"Detected programming language: {final_language}")
        
        return result
    
    def _detect_ambiguity(self, user_input: str) -> Dict[str, Any]:
        """
        Detect ambiguity in user input using pattern matching.
        
        Args:
            user_input: Natural language requirement
            
        Returns:
            Dictionary with ambiguity detection results
        """
        vague_terms = [
            r'\b(user-friendly|user friendly)\b',
            r'\b(fast|quick|quickly)\b',
            r'\b(good|better|best)\b',
            r'\b(easy|simple|easily)\b',
            r'\b(nice|nice-looking|pretty)\b',
            r'\b(some|various|multiple|several)\b',
            r'\b(should|could|might|may)\b',
        ]
        
        missing_patterns = [
            r'\b(input|output)\b',  # Check if input/output formats are mentioned
            r'\b(error|exception|handle)\b',  # Check if error handling is mentioned
            r'\b(platform|os|operating system)\b',  # Check if platform is specified
            r'\b(performance|speed|time)\b',  # Check if performance is mentioned
        ]
        
        vague_count = sum(1 for pattern in vague_terms if re.search(pattern, user_input, re.IGNORECASE))
        missing_count = sum(1 for pattern in missing_patterns if not re.search(pattern, user_input, re.IGNORECASE))
        
        is_ambiguous = vague_count > 2 or missing_count > 2 or len(user_input.strip()) < 50
        
        return {
            "is_ambiguous": is_ambiguous,
            "vague_terms_found": vague_count,
            "missing_specifications": missing_count,
            "input_length": len(user_input),
        }
    
    def _detect_programming_language(self, user_input: str) -> str:
        """
        Detect programming language from user input.
        
        Args:
            user_input: Natural language requirement
            
        Returns:
            Detected programming language (defaults to 'python')
        """
        user_lower = user_input.lower()
        
        # Language detection patterns
        language_patterns = {
            "python": [
                r'\bpython\b',
                r'\.py\b',
                r'\bpip\b',
                r'\bpyinstaller\b',
                r'\bdjango\b',
                r'\bflask\b',
                r'\bpytest\b',
            ],
            "react": [
                r'\breact\b',
                r'\bjsx\b',
                r'\.jsx\b',
                r'\.tsx\b',
                r'\breactjs\b',
                r'\breact\.js\b',
                r'\bcreate-react-app\b',
                r'\bnext\.js\b',
                r'\bgatsby\b',
            ],
            "javascript": [
                r'\bjavascript\b',
                r'\bjs\b',
                r'\.js\b',
                r'\bnpm\b',
                r'\bnode\.js\b',
                r'\bnodejs\b',
                r'\bpackage\.json\b',
                r'\bexpress\b',
            ],
            "typescript": [
                r'\btypescript\b',
                r'\bts\b',
                r'\.ts\b',
                r'\.tsx\b',
            ],
            "java": [
                r'\bjava\b',
                r'\.java\b',
                r'\bmaven\b',
                r'\bpom\.xml\b',
                r'\bgradle\b',
                r'\bspring\b',
            ],
            "cpp": [
                r'\bc\+\+\b',
                r'\bcpp\b',
                r'\.cpp\b',
                r'\.hpp\b',
                r'\bcmake\b',
            ],
            "csharp": [
                r'\bc#\b',
                r'\bcsharp\b',
                r'\.cs\b',
                r'\.net\b',
            ],
            "go": [
                r'\bgo\b',
                r'\bgolang\b',
                r'\.go\b',
                r'\bgo\.mod\b',
            ],
            "rust": [
                r'\brust\b',
                r'\.rs\b',
                r'\bcargo\b',
            ],
            "ruby": [
                r'\bruby\b',
                r'\.rb\b',
                r'\bgemfile\b',
                r'\brails\b',
            ],
            "php": [
                r'\bphp\b',
                r'\.php\b',
                r'\bcomposer\b',
            ],
            "swift": [
                r'\bswift\b',
                r'\.swift\b',
            ],
            "kotlin": [
                r'\bkotlin\b',
                r'\.kt\b',
            ],
        }
        
        # Check for explicit language mentions first
        # Priority order: React first (since it's a subset of JavaScript), then others
        priority_order = ["react", "typescript", "javascript"] + [lang for lang in language_patterns.keys() if lang not in ["react", "typescript", "javascript"]]
        
        for lang in priority_order:
            if lang in language_patterns:
                patterns = language_patterns[lang]
                for pattern in patterns:
                    if re.search(pattern, user_lower):
                        return lang
        
        # Default to Python if no language detected
        return "python"
    
    def _parse_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback parser if JSON extraction fails."""
        return {
            "functional_requirements": [content],
            "non_functional_requirements": [],
            "assumptions": ["Could not parse structured requirements - using raw input"],
            "constraints": [],
            "programming_language": "python",  # Default to Python
            "clarifying_questions": [
                {
                    "question": "Could not parse structured requirements",
                    "assumption": "Using raw input as requirement",
                    "code": "# JSON parsing failed - requirements may be incomplete"
                }
            ],
            "ambiguity_detected": True,
            "ambiguity_notes": "JSON parsing failed - requirements may be incomplete",
        }

