"""
Streamlit UI for Multi-Agent Coding Framework.
Provides interactive interface for user requirements and displays all agent outputs.
"""
import streamlit as st
import sys
import os
import logging
from orchestrator import Orchestrator
from utils.config import Config
from utils.logger import setup_logging, get_logger
from autogen import ConversableAgent

# Setup logging
setup_logging(
    level=logging.INFO,
    log_to_file=True,
    log_file_path="logs/streamlit_app.log"
)
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Coding Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: transparent;
        border: 1px solid #c3e6cb;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-icon-container {
        position: relative !important;
        display: inline-block !important;
    }
    .info-icon {
        font-size: 1.3rem !important;
        cursor: pointer !important;
        color: rgba(31, 119, 180, 0.6) !important;
        background-color: transparent !important;
        border-radius: 50% !important;
        width: 30px !important;
        height: 30px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        border: none !important;
        margin: 10px 0 !important;
        opacity: 0.7 !important;
    }
    .info-icon:hover {
        color: rgba(31, 119, 180, 1) !important;
        opacity: 1 !important;
        transform: scale(1.1);
    }
    .info-tooltip {
        position: absolute;
        top: 40px;
        left: 0;
        background-color: #2d3748;
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        min-width: 250px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s ease, visibility 0.2s ease;
        pointer-events: none;
        font-size: 0.9rem;
        line-height: 1.6;
        z-index: 1000000;
    }
    .info-icon-container:hover .info-tooltip {
        opacity: 1 !important;
        visibility: visible !important;
    }
    .info-tooltip::before {
        content: '';
        position: absolute;
        top: -6px;
        left: 20px;
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-bottom: 6px solid #2d3748;
    }
    .info-tooltip-item {
        margin: 8px 0;
    }
    .info-tooltip-label {
        font-weight: bold;
        color: #a0aec0;
        margin-right: 8px;
    }
    </style>
""", unsafe_allow_html=True)


def detect_follow_up(new_prompt: str, previous_context: dict) -> bool:
    """
    Detect if a new prompt is a follow-up to previous conversation or a new prompt.
    
    Args:
        new_prompt: The new user prompt
        previous_context: Dictionary containing previous prompts and results
        
    Returns:
        True if it's a follow-up, False if it's a new prompt
    """
    if not previous_context.get("is_active") or not previous_context.get("previous_prompts"):
        return False
    
    previous_prompt = previous_context["previous_prompts"][-1]
    
    # Use LLM to detect if it's a follow-up
    try:
        agent = ConversableAgent(
            name="followup_detector",
            system_message="""You are a prompt classifier. Your task is to determine if a new user prompt is a follow-up to a previous conversation or a completely new request.

A follow-up prompt:
- References or modifies the previous request
- Asks for changes, updates, or modifications to previously generated code
- Continues the same topic or project
- Uses words like "change", "update", "modify", "add", "remove", "instead", "also", "also add", "make it", etc.
- References previous context implicitly

A new prompt:
- Is completely unrelated to the previous request
- Starts a new topic or project
- Doesn't reference anything from the previous conversation
- Is about a different software/project entirely

Respond with ONLY "FOLLOWUP" or "NEW" (no other text).""",
            llm_config={
                "config_list": [{
                    "model": Config.MODEL,
                    "api_key": Config.OPENAI_API_KEY,
                    "temperature": 0.3,
                }],
                "timeout": 30,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        
        prompt = f"""PREVIOUS PROMPT:
{previous_prompt}

NEW PROMPT:
{new_prompt}

Is the new prompt a follow-up to the previous prompt or a completely new request?
Respond with ONLY "FOLLOWUP" or "NEW"."""
        
        response = agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        
        if response:
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            content = content.strip().upper()
            return "FOLLOWUP" in content or "FOLLOW-UP" in content
        
        # Fallback to heuristic if LLM fails
        return _heuristic_followup_detection(new_prompt, previous_prompt)
        
    except Exception as e:
        logger.warning(f"Follow-up detection failed: {str(e)}, using heuristic")
        return _heuristic_followup_detection(new_prompt, previous_prompt)


def _heuristic_followup_detection(new_prompt: str, previous_prompt: str) -> bool:
    """Heuristic-based fallback for follow-up detection."""
    followup_keywords = [
        "change", "update", "modify", "add", "remove", "also", "instead", 
        "make it", "can you", "please", "the code", "the previous", 
        "above", "that", "it", "this", "same", "keep", "maintain"
    ]
    
    new_lower = new_prompt.lower()
    prev_lower = previous_prompt.lower()
    
    # Check for follow-up keywords
    keyword_count = sum(1 for keyword in followup_keywords if keyword in new_lower)
    
    # If very short, likely a follow-up
    if len(new_prompt.strip()) < 50 and keyword_count > 0:
        return True
    
    # If contains multiple follow-up keywords, likely a follow-up
    if keyword_count >= 2:
        return True
    
    # If explicitly references previous context
    if any(ref in new_lower for ref in ["previous", "above", "that code", "the code", "same"]):
        return True
    
    return False


def initialize_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        try:
            # Force reload config to get latest model settings
            import importlib
            import utils.config
            importlib.reload(utils.config)
            st.session_state.orchestrator = Orchestrator()
        except ValueError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")
            st.stop()
    
    if "results" not in st.session_state:
        st.session_state.results = None
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    
    # Conversation context for follow-up prompts
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {
            "previous_prompts": [],
            "previous_results": None,
            "is_active": False
        }
    
    # Input key for clearing text area
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    # Instructions expander state
    if "instructions_expanded" not in st.session_state:
        st.session_state.instructions_expanded = True
    
    # Flag to track if generate was just clicked (for immediate stop button enable)
    if "generate_clicked" not in st.session_state:
        st.session_state.generate_clicked = False


def display_requirements(results: dict):
    """Display requirement analysis results."""
    st.subheader("📋 Requirement Analysis")
    
    requirements = results.get("requirements", {})
    
    # Display ambiguity information if available
    ambiguity_detected = requirements.get("ambiguity_detected", False)
    clarifying_questions = requirements.get("clarifying_questions", [])
    ambiguity_notes = requirements.get("ambiguity_notes", "")
    
    if ambiguity_detected or clarifying_questions:
        with st.expander("🔍 Ambiguity Detection & Clarifying Questions", expanded=True):
            if ambiguity_detected:
                st.warning("⚠️ Ambiguity detected in requirements")
            if ambiguity_notes:
                st.info(f"**Notes:** {ambiguity_notes}")
            if clarifying_questions:
                st.markdown("**Clarifying Questions Generated:**")
                for i, q_item in enumerate(clarifying_questions, 1):
                    # Handle both old format (string) and new format (dict)
                    if isinstance(q_item, dict):
                        question = q_item.get("question", "")
                        assumption = q_item.get("assumption", "")
                        code = q_item.get("code", "")
                    else:
                        # Old format - backward compatibility
                        question = q_item
                        assumption = ""
                        code = ""
                    
                    # Display question
                    st.markdown(f"**{i}. {question}**")
                    
                    # Display assumption if available
                    if assumption:
                        with st.container():
                            st.markdown("**Assumption Made:**")
                            st.info(assumption)
                    
                    # Display code if available
                    if code:
                        with st.container():
                            st.markdown("**Code Generated:**")
                            st.code(code, language="python")
                    
                    # Add spacing between questions
                    if i < len(clarifying_questions):
                        st.divider()
            else:
                st.info("No specific clarifying questions generated")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Functional Requirements:**")
        func_reqs = requirements.get("functional_requirements", [])
        if func_reqs:
            for req in func_reqs:
                st.markdown(f"- {req}")
        else:
            st.info("No functional requirements extracted")
        
        st.markdown("**Assumptions:**")
        assumptions = requirements.get("assumptions", [])
        if assumptions:
            for assumption in assumptions:
                st.markdown(f"- {assumption}")
        else:
            st.info("No assumptions identified")
    
    with col2:
        st.markdown("**Non-Functional Requirements:**")
        non_func_reqs = requirements.get("non_functional_requirements", [])
        if non_func_reqs:
            for req in non_func_reqs:
                st.markdown(f"- {req}")
        else:
            st.info("No non-functional requirements extracted")
        
        st.markdown("**Constraints:**")
        constraints = requirements.get("constraints", [])
        if constraints:
            for constraint in constraints:
                st.markdown(f"- {constraint}")
        else:
            st.info("No constraints identified")


def _get_language_from_filename(filename: str) -> str:
    """
    Determine programming language from file extension.
    
    Args:
        filename: File name with extension
        
    Returns:
        Language identifier for syntax highlighting
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'jsx': 'javascript',  # JSX files use JavaScript syntax highlighting
        'ts': 'typescript',
        'tsx': 'typescript',
        'java': 'java',
        'cpp': 'cpp',
        'cc': 'cpp',
        'cxx': 'cpp',
        'c': 'c',
        'cs': 'csharp',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'swift': 'swift',
        'kt': 'kotlin',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'xml': 'xml',
        'sql': 'sql',
        'sh': 'bash',
        'bash': 'bash',
    }
    
    return language_map.get(ext, 'python')  # Default to Python


def _get_mime_type_from_filename(filename: str) -> str:
    """
    Get MIME type from file extension.
    
    Args:
        filename: File name with extension
        
    Returns:
        MIME type string
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    mime_map = {
        'py': 'text/x-python',
        'js': 'text/javascript',
        'ts': 'text/typescript',
        'tsx': 'text/typescript',
        'java': 'text/x-java',
        'cpp': 'text/x-c++',
        'cc': 'text/x-c++',
        'cxx': 'text/x-c++',
        'c': 'text/x-c',
        'cs': 'text/x-csharp',
        'go': 'text/x-go',
        'rs': 'text/x-rust',
        'rb': 'text/x-ruby',
        'php': 'text/x-php',
        'swift': 'text/x-swift',
        'kt': 'text/x-kotlin',
        'html': 'text/html',
        'css': 'text/css',
        'json': 'application/json',
        'xml': 'application/xml',
        'sql': 'text/x-sql',
        'sh': 'text/x-shellscript',
        'bash': 'text/x-shellscript',
    }
    
    return mime_map.get(ext, 'text/plain')


def _parse_multiple_files(code: str, language: str = "python") -> list:
    """
    Parse code string to detect multiple files.
    Looks for patterns like:
    - # File: filename.ext
    - ## File: filename.ext
    - ## filename.ext
    - # filename.ext
    
    Args:
        code: Code string to parse
        language: Programming language (default: "python") - used for default filename if no files detected
    
    Returns:
        List of dicts with 'filename' and 'content' keys.
    """
    import re
    
    if not code or not code.strip():
        # Use appropriate extension based on language
        ext_map = {
            "python": "py",
            "javascript": "js",
            "react": "jsx",  # React uses JSX extension
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "csharp": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php",
            "swift": "swift",
            "kotlin": "kt",
        }
        ext = ext_map.get(language.lower(), "py")
        return [{"filename": f"generated_code.{ext}", "content": code}]
    
    files = []
    
    # Strategy: Split by file markers and extract content between them
    # This is more reliable than regex lookaheads
    
    # Find all file markers with their positions
    file_markers = []
    
    # Pattern 1: "# File: filename.ext" or "# File:filename.ext" (supports any extension)
    pattern1 = r'^#\s*File:\s*([^\n]+\.[a-zA-Z0-9]+)\s*$'
    for match in re.finditer(pattern1, code, re.MULTILINE):
        file_markers.append({
            'pos': match.start(),
            'filename': match.group(1).strip(),
            'line_end': match.end()
        })
    
    # Pattern 2: "## File: filename.ext" or "## filename.ext" or "# filename.ext" (supports any extension)
    if not file_markers:
        pattern2 = r'^#+\s*(?:File:\s*)?([^\n]+\.[a-zA-Z0-9]+)\s*$'
        for match in re.finditer(pattern2, code, re.MULTILINE):
            file_markers.append({
                'pos': match.start(),
                'filename': match.group(1).strip(),
                'line_end': match.end()
            })
    
    # If we found file markers, extract content between them
    if file_markers:
        # Sort by position
        file_markers.sort(key=lambda x: x['pos'])
        
        for i, marker in enumerate(file_markers):
            filename = marker['filename']
            # Start from after the marker line (skip the marker line itself)
            start_pos = marker['line_end']
            
            # Skip only newlines/carriage returns after the marker (preserve spaces/tabs for indentation)
            while start_pos < len(code) and code[start_pos] in ['\n', '\r']:
                start_pos += 1
            
            # Find end position (start of next file marker or end of code)
            if i + 1 < len(file_markers):
                end_pos = file_markers[i + 1]['pos']
                # Move back to exclude only trailing newlines/carriage returns before next marker
                while end_pos > start_pos and code[end_pos - 1] in ['\n', '\r']:
                    end_pos -= 1
            else:
                end_pos = len(code)
            
            # Extract content between this marker and next marker
            file_content = code[start_pos:end_pos]
            
            # Clean up: remove only leading/trailing newlines but preserve all other content including indentation
            file_content = file_content.strip('\n\r')
            
            if filename and file_content:
                files.append({
                    "filename": filename,
                    "content": file_content
                })
    
    # Fallback: Line-by-line parsing if regex didn't work
    if not files:
        lines = code.split('\n')
        current_file = None
        current_content = []
        
        for line in lines:
            # Check for file markers at line start (supports any file extension)
            file_match = re.match(r'^#+\s*(?:File:\s*)?([^\s:]+\.[a-zA-Z0-9]+)\s*$', line.strip())
            if file_match:
                # Save previous file if exists
                if current_file and current_content:
                    content_str = '\n'.join(current_content).strip()
                    if content_str:
                        files.append({
                            "filename": current_file,
                            "content": content_str
                        })
                # Start new file
                current_file = file_match.group(1)
                current_content = []
            else:
                if current_file:
                    current_content.append(line)
                else:
                    # Collect content before first file marker (if any)
                    current_content.append(line)
        
        # Add last file
        if current_file and current_content:
            content_str = '\n'.join(current_content).strip()
            if content_str:
                files.append({
                    "filename": current_file,
                    "content": content_str
                })
    
    # If still no files found, treat entire code as single file
    if not files:
        # Use appropriate extension based on language
        ext_map = {
            "python": "py",
            "javascript": "js",
            "react": "jsx",  # React uses JSX extension
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "csharp": "cs",
            "go": "go",
            "rust": "rs",
            "ruby": "rb",
            "php": "php",
            "swift": "swift",
            "kotlin": "kt",
        }
        ext = ext_map.get(language.lower(), "py")
        files = [{
            "filename": f"generated_code.{ext}",
            "content": code.strip()
        }]
    
    return files


def display_code(results: dict):
    """Display generated code, showing multiple files separately if detected."""
    st.subheader("💻 Generated Code")
    
    code = results.get("code", "")
    requirements = results.get("requirements", {})
    
    # Get programming language from requirements
    language = requirements.get("programming_language", "python").lower() if requirements else "python"
    
    if code:
        # Parse for multiple files (pass language for default filename)
        files = _parse_multiple_files(code, language)
        
        if len(files) > 1:
            # Multiple files detected - show each separately
            st.info(f"📁 Detected {len(files)} files. Displaying each file separately:")
            
            for idx, file_info in enumerate(files, 1):
                filename = file_info["filename"]
                file_content = file_info["content"]
                
                # Determine language for syntax highlighting
                code_language = _get_language_from_filename(filename)
                mime_type = _get_mime_type_from_filename(filename)
                
                # Create a container for each file
                with st.container():
                    st.markdown(f"#### 📄 File {idx}: `{filename}`")
                    st.code(file_content, language=code_language)
                    
                    # Download button for each file
                    st.download_button(
                        label=f"📥 Download {filename}",
                        data=file_content,
                        file_name=filename,
                        mime=mime_type,
                        key=f"download_{filename}_{idx}"
                    )
                    
                    # Add separator between files (except for last one)
                    if idx < len(files):
                        st.divider()
        else:
            # Single file - display as before
            file_info = files[0]
            filename = file_info["filename"]
            file_content = file_info["content"]
            
            # Determine language for syntax highlighting
            code_language = _get_language_from_filename(filename)
            mime_type = _get_mime_type_from_filename(filename)
            
            st.code(file_content, language=code_language)
            
            # Download button for code
            st.download_button(
                label=f"📥 Download {filename}",
                data=file_content,
                file_name=filename,
                mime=mime_type
            )
    else:
        st.error("No code generated")


def display_review_feedback(results: dict):
    """Display code review feedback."""
    st.subheader("🔍 Code Review")
    
    feedbacks = results.get("review_feedback", [])
    iterations = results.get("iterations", 0)
    
    st.info(f"Total Iterations: {iterations}")
    
    if feedbacks:
        for i, feedback in enumerate(feedbacks, 1):
            with st.expander(f"Iteration {i} Feedback", expanded=(i == len(feedbacks))):
                if feedback.upper().startswith("APPROVED"):
                    st.success(feedback)
                else:
                    st.warning(feedback)
    else:
        st.info("No review feedback available")


def display_documentation(results: dict):
    """Display generated documentation."""
    st.subheader("📚 Documentation")
    
    documentation = results.get("documentation", "")
    if documentation:
        # Show table of contents indicator
        if "##" in documentation or "#" in documentation:
            st.markdown("#### 📑 Documentation Structure")
            st.success("✅ Documentation includes: Overview, Agent Overview, Workflow, Setup, Usage, API Reference, and Examples")
        
        # Display documentation with better formatting
        st.markdown(documentation)
        
        # Download button for documentation
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Documentation",
                data=documentation,
                file_name="documentation.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.info("No documentation generated")


def display_test_cases(results: dict):
    """Display generated test cases (unit and integration tests)."""
    st.subheader("🧪 Test Cases")
    
    test_cases = results.get("test_cases", "")
    
    if test_cases:
        # Parse test cases to separate unit and integration tests
        unit_tests, integration_tests = _parse_test_types(test_cases)
        
        # Display Unit Tests
        if unit_tests:
            st.markdown("#### 🔬 Unit Tests")
            st.info("Unit tests test individual functions, methods, and classes in isolation.")
            with st.expander("📝 View Unit Test Code", expanded=True):
                st.code(unit_tests, language="python")
            
            st.download_button(
                label="📥 Download Unit Tests",
                data=unit_tests,
                file_name="test_unit.py",
                mime="text/x-python",
                key="download_unit_tests"
            )
        
        # Display Integration Tests
        if integration_tests:
            if unit_tests:
                st.divider()
            st.markdown("#### 🔗 Integration Tests")
            st.info("Integration tests test how multiple components work together.")
            with st.expander("📝 View Integration Test Code", expanded=True):
                st.code(integration_tests, language="python")
            
            st.download_button(
                label="📥 Download Integration Tests",
                data=integration_tests,
                file_name="test_integration.py",
                mime="text/x-python",
                key="download_integration_tests"
            )
        
        # Download all tests together
        if unit_tests or integration_tests:
            st.divider()
            st.download_button(
                label="📥 Download All Test Cases",
                data=test_cases,
                file_name="test_generated_code.py",
                mime="text/x-python",
                key="download_all_tests"
            )
    else:
        st.info("No test cases generated")


def _parse_test_types(test_cases: str) -> tuple:
    """
    Parse test cases to separate unit tests and integration tests.
    
    Args:
        test_cases: Combined test code string
        
    Returns:
        Tuple of (unit_tests, integration_tests)
    """
    import re
    
    if not test_cases or not test_cases.strip():
        return "", ""
    
    # Look for section markers (most common patterns)
    unit_patterns = [
        r'^#\s*Unit\s+Tests?',
        r'^#\s*UNIT\s+TESTS?',
        r'^#\s*Unit\s+Test',
    ]
    
    integration_patterns = [
        r'^#\s*Integration\s+Tests?',
        r'^#\s*INTEGRATION\s+TESTS?',
        r'^#\s*Integration\s+Test',
    ]
    
    lines = test_cases.split('\n')
    unit_start = None
    integration_start = None
    
    # Find section markers
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Check for unit test markers
        if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in unit_patterns):
            if unit_start is None:
                unit_start = i
        # Check for integration test markers
        if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in integration_patterns):
            if integration_start is None:
                integration_start = i
    
    # If we found both markers, split accordingly
    if unit_start is not None and integration_start is not None:
        if unit_start < integration_start:
            # Unit tests come first
            unit_tests = '\n'.join(lines[unit_start:integration_start])
            integration_tests = '\n'.join(lines[integration_start:])
        else:
            # Integration tests come first
            unit_tests = '\n'.join(lines[unit_start:])
            integration_tests = '\n'.join(lines[integration_start:unit_start])
        return unit_tests.strip(), integration_tests.strip()
    elif unit_start is not None:
        # Only unit tests marker found - everything after it is unit tests
        unit_tests = '\n'.join(lines[unit_start:])
        return unit_tests.strip(), ""
    elif integration_start is not None:
        # Only integration tests marker found - everything after it is integration tests
        integration_tests = '\n'.join(lines[integration_start:])
        return "", integration_tests.strip()
    else:
        # No clear markers found - return all as unit tests (default)
        # The agent should be generating markers, but if not, we default to unit tests
        return test_cases.strip(), ""


def display_deployment_config(results: dict):
    """Display deployment configuration."""
    st.subheader("🚀 Deployment Configuration")
    
    deployment = results.get("deployment_config", {})
    
    if deployment:
        # Requirements.txt
        st.markdown("**📦 requirements.txt:**")
        requirements = deployment.get("requirements", "")
        if requirements:
            st.code(requirements, language="text")
            st.download_button(
                label="📥 Download requirements.txt",
                data=requirements,
                file_name="requirements.txt",
                mime="text/plain"
            )
        
        st.divider()
        
        # Setup Instructions
        st.markdown("**⚙️ Project Setup Instructions:**")
        setup = deployment.get("setup_instructions", "")
        if setup:
            st.markdown(setup)
        
        st.divider()
        
        # GitHub Push Instructions
        st.markdown("**🔗 GitHub Push Instructions:**")
        github_push = deployment.get("github_push", "")
        if github_push:
            st.markdown(github_push)
        else:
            st.info("GitHub push instructions not available")
        
        st.divider()
        
        # Hosting Platform Recommendations
        st.markdown("**🌐 Hosting Platform Recommendations:**")
        hosting_platforms = deployment.get("hosting_platforms", "")
        if hosting_platforms:
            st.markdown(hosting_platforms)
        else:
            st.info("Hosting platform recommendations not available")
    else:
        st.info("No deployment configuration generated")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Info icon at top left using columns for reliable positioning
    top_col1, top_col2 = st.columns([0.08, 0.92])
    with top_col1:
        st.markdown(f"""
        <div class="info-icon-container">
            <div class="info-icon">ℹ️</div>
            <div class="info-tooltip">
                <div class="info-tooltip-item">
                    <span class="info-tooltip-label">Model:</span>{Config.MODEL}
                </div>
                <div class="info-tooltip-item">
                    <span class="info-tooltip-label">Max Iterations:</span>{Config.MAX_ITERATIONS}
                </div>
                <div class="info-tooltip-item">
                    <span class="info-tooltip-label">Framework:</span>Multi-Agent Coding Framework v1.0.0
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_col2:
        st.markdown('<div class="main-header">🤖 Multi-Agent Coding Framework</div>', unsafe_allow_html=True)
    
    # Main content area
    st.header("📝 Enter Requirements")
    
    # Input validation helper
    with st.expander("ℹ️  How to Write Good Requirements", expanded=False):
        st.markdown("""
        **Tips for best results:**
        - Be specific about functionality needed
        - Mention any constraints or requirements
        - Include examples if helpful
        - Specify input/output formats if relevant
        
        **Example:**
        ```
        Create a Python calculator that can perform basic arithmetic operations 
        (addition, subtraction, multiplication, division) with error handling 
        for division by zero. The calculator should accept two numbers and an 
        operation as input, and return the result.
        ```
        """)
    
    user_input = st.text_area(
        "Describe your software requirements:",
        height=150,
        placeholder="Example: Create a Python calculator that can perform basic arithmetic operations (addition, subtraction, multiplication, division) with error handling for division by zero.",
        help="Enter detailed requirements for the software you want to generate. Be as specific as possible for best results.",
        key=f"user_input_{st.session_state.input_key}"
    )
    
    # Input validation
    input_valid = True
    if user_input:
        if len(user_input.strip()) < 10:
            st.warning("⚠️  Input is very short. Please provide more detailed requirements for better results.")
        if len(user_input) > 5000:
            st.error("❌ Input is too long. Please keep requirements under 5000 characters.")
            input_valid = False
    
    col1, col2 = st.columns([1, 1])
    with col1:
        generate_button = st.button("🚀 Generate Code", type="primary", use_container_width=True, disabled=not input_valid or st.session_state.get("processing", False))
        # If generate button was clicked, set flag immediately
        if generate_button:
            st.session_state.generate_clicked = True
            st.session_state.processing = True
    with col2:
        # Stop button should be enabled if processing is True OR if generate was just clicked
        # This ensures it works even during follow-up prompts
        processing_state = st.session_state.get("processing", False) or st.session_state.get("generate_clicked", False)
        stop_button = st.button("⏹️ Stop", use_container_width=True, disabled=not processing_state)
        if stop_button:
            st.session_state.stop_requested = True
            st.session_state.processing = False
            st.session_state.generate_clicked = False
            st.warning("⏹️ Stop requested. Execution will stop after current step completes.")
            st.rerun()
    
    # Instructions section (hidden when processing)
    if not st.session_state.get("processing", False):
        with st.expander("📖 Instructions", expanded=st.session_state.get("instructions_expanded", True)):
            st.markdown("""
            **How to use:**
            1. Enter your software requirements in natural language
            2. Click 'Generate Code' to start the multi-agent pipeline
            3. Review outputs from each agent:
               - Requirement Analysis
               - Generated Code
               - Code Review Feedback
               - Documentation
               - Test Cases
               - Deployment Configuration
            4. Download any generated files as needed
            """)
    
    # Clear Results button (only show when there are results)
    if st.session_state.results:
        if st.button("🔄 Clear Results", use_container_width=False):
            st.session_state.results = None
            st.session_state.conversation_context = {
                "previous_prompts": [],
                "previous_results": None,
                "is_active": False
            }
            st.session_state.input_key += 1
            st.rerun()
    
    # Process user input
    if generate_button or st.session_state.get("generate_clicked", False):
        # Reset the flag
        st.session_state.generate_clicked = False
        
        if not user_input or not user_input.strip():
            st.error("❌ Please enter your requirements before generating code.")
            st.session_state.processing = False
        elif len(user_input.strip()) < 10:
            st.warning("⚠️  Requirements are too short. Please provide more details for better results.")
            st.session_state.processing = False
        else:
            # Ensure processing state is set (already set when button was clicked)
            st.session_state.processing = True
            st.session_state.stop_requested = False
            # Auto-minimize instructions tab
            st.session_state.instructions_expanded = False
            
            # Detect if this is a follow-up or new prompt
            is_followup = False
            context = None
            
            if st.session_state.conversation_context.get("is_active"):
                is_followup = detect_follow_up(
                    user_input, 
                    st.session_state.conversation_context
                )
                
                if is_followup:
                    logger.info("Detected follow-up prompt")
                    context = st.session_state.conversation_context
                else:
                    logger.info("Detected new prompt - resetting context")
                    st.session_state.conversation_context = {
                        "previous_prompts": [],
                        "previous_results": None,
                        "is_active": False
                    }
            
            # If new prompt, reset results
            if not is_followup:
                st.session_state.results = None
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(progress: int, message: str):
                """Update progress bar and status text."""
                progress_bar.progress(progress)
                if progress < 100:
                    status_text.info(message)
                else:
                    status_text.success(message)
            
            # Stop check callback function
            def check_stop() -> bool:
                """Check if stop was requested."""
                return st.session_state.get("stop_requested", False)
            
            try:
                with st.spinner("🤖 Agents are working... This may take a few minutes."):
                    results = st.session_state.orchestrator.execute_pipeline(
                        user_input, 
                        progress_callback=update_progress,
                        stop_check=check_stop,
                        context=context
                    )
                    
                    st.session_state.results = results
                    st.session_state.processing = False
                    st.session_state.stop_requested = False
                    
                    # Update conversation context
                    if not st.session_state.conversation_context.get("previous_prompts"):
                        st.session_state.conversation_context["previous_prompts"] = []
                    st.session_state.conversation_context["previous_prompts"].append(user_input)
                    st.session_state.conversation_context["previous_results"] = results
                    st.session_state.conversation_context["is_active"] = True
                    
                    # Clear input box by incrementing key
                    st.session_state.input_key += 1
                    
                    # Check if execution was stopped
                    if results.get("status") == "stopped":
                        st.warning("⏹️ Execution stopped by user. Partial results are shown below.")
                    
                    # Small delay to show completion
                    import time
                    time.sleep(0.5)
                    st.rerun()
            except ValueError as e:
                st.error(f"❌ Input Error: {str(e)}")
                st.info("💡 Tip: Make sure your requirements are clear and specific.")
                st.session_state.processing = False
                st.session_state.stop_requested = False
                progress_bar.empty()
                status_text.empty()
            except KeyboardInterrupt:
                st.warning("⏹️ Execution interrupted by user.")
                st.session_state.processing = False
                st.session_state.stop_requested = False
                progress_bar.empty()
                status_text.empty()
            except Exception as e:
                st.error(f"❌ Error during pipeline execution: {str(e)}")
                with st.expander("🔍 View Error Details"):
                    st.exception(e)
                st.session_state.processing = False
                st.session_state.stop_requested = False
                progress_bar.empty()
                status_text.empty()
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        status = results.get("status", "unknown")
        
        if status == "completed":
            st.markdown('<div class="success-box">✅ Pipeline execution completed successfully!</div>', unsafe_allow_html=True)
        elif status == "stopped":
            st.markdown('<div class="error-box">⏹️ Pipeline execution stopped by user. Partial results are shown below.</div>', unsafe_allow_html=True)
        elif status == "failed":
            st.markdown('<div class="error-box">❌ Pipeline execution failed. Please check the error messages below.</div>', unsafe_allow_html=True)
        elif status == "error":
            st.markdown('<div class="error-box">❌ An error occurred during pipeline execution.</div>', unsafe_allow_html=True)
            if "error" in results:
                st.error(f"Error: {results['error']}")
        
        # Display all agent outputs (show partial results if stopped)
        if status in ["completed", "stopped"]:
            st.divider()
            if results.get("requirements"):
                display_requirements(results)
                st.divider()
            if results.get("code"):
                display_code(results)
                st.divider()
            if results.get("review_feedback"):
                display_review_feedback(results)
                st.divider()
            if results.get("documentation"):
                display_documentation(results)
                st.divider()
            if results.get("test_cases"):
                display_test_cases(results)
                st.divider()
            if results.get("deployment_config"):
                display_deployment_config(results)


if __name__ == "__main__":
    main()

