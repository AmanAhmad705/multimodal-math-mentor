"""
Streamlit UI: Main application interface
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.orchestrator import MathMentorOrchestrator
from src.memory.store import MemoryStore

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Math Mentor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .trace-step { padding: 0.5rem; margin: 0.25rem 0; border-left: 4px solid #1f77b4; }
    .trace-success { border-left-color: #2ca02c; }
    .trace-warning { border-left-color: #ff7f0e; }
    .trace-error { border-left-color: #d62728; }
    .solution-card { padding: 1rem; background: #f0f2f6; border-radius: 0.5rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state"""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = MathMentorOrchestrator()
        st.session_state.memory_store = MemoryStore()
        st.session_state.current_result = None
        st.session_state.hitl_pending = False
        st.session_state.hitl_id = None

def display_trace(trace_dict):
    """Display execution trace"""
    st.markdown("### Execution Trace")
    with st.expander("Show detailed trace"):
        for step in trace_dict.get("steps", []):
            css_class = f"trace-step trace-{step.get('status', 'success')}"
            st.markdown(
                f"""
                <div class="{css_class}">
                    <b>{step.get('agent', '')}</b>: {step.get('action', '')}<br/>
                    {step.get('result', '')}
                </div>
                """,
                unsafe_allow_html=True
            )

def display_solution(result):
    """Display solution components"""
    if result.get("status") != "success":
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem", "Solution", "Explanation", "Context", "Trace"
    ])

    with tab1:
        st.markdown("### Parsed Problem")
        problem = result.get("parsed_problem", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Topic", problem.get("topic", "N/A"))
        with col2:
            st.metric("Difficulty", problem.get("difficulty", "N/A"))
        with col3:
            st.metric("Variables", len(problem.get("variables", [])))
        
        st.markdown("#### Problem Statement")
        st.write(problem.get("text", ""))

        if problem.get("variables"):
            st.markdown("#### Variables")
            st.write(", ".join(problem["variables"]))

        if problem.get("constraints"):
            st.markdown("#### Constraints")
            st.write(", ".join(problem["constraints"]))

    with tab2:
        st.markdown("### Solution")
        solution = result.get("solution", {})

        st.markdown("#### Step-by-Step Solution")
        for i, step in enumerate(solution.get("steps", []), 1):
            st.write(f"**Step {i}:** {step}")

        st.markdown("#### Final Answer")
        st.info(solution.get("final_answer", "N/A"))

        # Key formulas used (optional)
        formulas = solution.get("key_formulas_used", [])
        if formulas:
            st.markdown("#### Key Formulas Used")
            for formula in formulas:
                st.write(f"• {formula}")

        st.markdown("#### Solution Confidence")
        st.progress(solution.get("confidence", 0))

        # Verification
        verification = result.get("verification", {})
        status = "✓ Verified" if verification.get("is_correct") else "⚠ Not Verified"
        st.markdown("#### Verification Result")
        st.write(f"**Status:** {status}")
        st.write(f"**Confidence:** {verification.get('confidence', 0):.2%}")

        issues = verification.get("issues", [])
        if issues:
            st.warning("**Issues Found:**")
            for issue in issues:
                st.write(f"• {issue}")

    with tab3:
        st.markdown("### Student-Friendly Explanation")
        explanation = result.get("explanation", {})
        st.markdown("#### Conceptual Overview")
        st.write(explanation.get("overview", ""))

        st.markdown("#### Step-by-Step Breakdown")
        for i, step in enumerate(explanation.get("steps", []), 1):
            st.write(f"**{i}.** {step}")

        st.markdown("#### Key Insights")
        for insight in explanation.get("insights", []):
            st.write(f"• {insight}")

        st.markdown("#### Common Mistakes")
        for mistake in explanation.get("mistakes", []):
            st.write(f"⚠ {mistake}")

        st.markdown("#### Related Concepts")
        for concept in explanation.get("related", []):
            st.write(f"🔗 {concept}")

    with tab4:
        st.markdown("### Retrieved Context (RAG)")
        context = result.get("retrieved_context", [])
        if context:
            for i, doc in enumerate(context, 1):
                with st.expander(f"[{i}] {doc.get('title', 'Untitled')} ({doc.get('relevance', 0):.0%} relevant)"):
                    st.write(doc.get("content", ""))
        else:
            st.info("No specific knowledge base documents retrieved.")

    with tab5:
        st.markdown("### Execution Trace")
        trace = result.get("trace", {})
        if trace:
            for step in trace.get("steps", []):
                css_class = f"trace-step trace-{step.get('status', 'success')}"
                st.markdown(
                    f'<div class="{css_class}"><b>{step.get("agent")}</b>: {step.get("action")}<br/>{step.get("result")}</div>',
                    unsafe_allow_html=True,
                )


def handle_hitl(hitl_id: str, content: str):
    """Handle HITL request"""
    st.warning("🔔 Human-in-the-Loop Review Required")
    st.write(f"HITL ID: {hitl_id}")
    st.json(content)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Approve"):
            return {"decision": "approved", "feedback": "User approved"}
    with col2:
        if st.button("Reject"):
            feedback = st.text_input("Reason for rejection:")
            if feedback:
                return {"decision": "rejected", "feedback": feedback}
    with col3:
        if st.button("Correct"):
            correction = st.text_area("Corrected version:")
            if correction:
                return {"decision": "correction", "corrected_text": correction}
    
    return None

def main():
    """Main application"""
    initialize_session()
    
    # Header
    st.title("📐 Multimodal Math Mentor")
    st.markdown(
        "Solve JEE-style math problems with step-by-step explanations, "
        "RAG-powered context, and intelligent verification."
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown(f"**Session ID:** `{st.session_state.orchestrator.session_id[:8]}...`")
        
        if st.button("New Session"):
            st.session_state.orchestrator = MathMentorOrchestrator()
            st.rerun()
        
        st.markdown("---")
        st.markdown("## Statistics")
        stats = st.session_state.memory_store.get_session_statistics(
            st.session_state.orchestrator.session_id
        )
        st.metric("Problems Solved", stats.get("problems_solved",0))
        st.metric("Avg Confidence", f"{stats.get('average_confidence',0.0):.2%}")
        st.metric("HITL Triggers", stats.get("hitl_triggers",0))
    
    # Main interface
    tab_input, tab_history = st.tabs(["📝 New Problem", "📊 Session History"])
    
    with tab_input:
        st.markdown("### Input Method")
        input_method = st.radio(
            "Select input method:",
            ["Text", "Image", "Audio"],
            horizontal=True,
        )
        
        raw_result = None
        input_id = None
        
        if input_method == "Text":
            st.markdown("#### Type or Paste Problem")
            problem_text = st.text_area(
                "Problem statement:",
                placeholder="e.g., Solve for x: x^2 - 5x + 6 = 0",
                height=120,
            )
            if problem_text and st.button("Parse & Solve"):
                raw_result = st.session_state.orchestrator.process_text(problem_text)
                input_id = raw_result.get("input_id")
        
        elif input_method == "Image":
            st.markdown("#### Upload Image")
            uploaded_file = st.file_uploader(
                "Upload image (JPG/PNG):",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                st.image(uploaded_file, caption="Uploaded image")
                
                if st.button("Extract & Solve"):
                    with st.spinner("Extracting text from image..."):
                        raw_result = st.session_state.orchestrator.process_image(tmp_path)
                        input_id = raw_result.get("input_id")
                    
                    if raw_result.get("needs_hitl"):
                        st.warning(
                            f"OCR confidence low ({raw_result.get('confidence',0.0):.0%}). "
                            "Please review extracted text."
                        )
                        extracted_text = st.text_area(
                            "Edit extracted text:",
                            value=raw_result.get("raw_text",""),
                            height=150,
                        )
                        if st.button("Use Edited Text"):
                            raw_result["raw_text"] = extracted_text
        
        elif input_method == "Audio":
            st.markdown("#### Upload Audio")

            uploaded_file = st.file_uploader(
                "Upload audio (MP3/WAV):",
                type=["mp3", "wav", "m4a"],
            )

            if uploaded_file is not None:
                # ✅ Windows-safe persistent temp file
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, f"audio_{uploaded_file.name}")

                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.audio(uploaded_file)

                if st.button("Transcribe & Solve"):
                    with st.spinner("Transcribing audio..."):
                        raw_result = st.session_state.orchestrator.process_audio(tmp_path)
                        input_id = raw_result.get("input_id")

                    if raw_result.get("needs_hitl"):
                        st.warning(
                            f"⚠️ ASR confidence low ({raw_result['confidence']:.0%}). "
                            "Please review transcript."
                        )
                        transcript = st.text_area(
                            "Edit transcript:",
                            value=raw_result["raw_text"],
                            height=150,
                        )
                        if st.button("Use Edited Transcript"):
                            raw_result["raw_text"] = transcript


        
        # Solve
        if raw_result and input_id:
            with st.spinner("Solving problem..."):
                result = st.session_state.orchestrator.solve_problem(
                    raw_result.get("raw_text", ""),
                    input_id,
                    handle_hitl_callback=handle_hitl,
                )
            
            st.session_state.current_result = result
            
            status = result.get("status")

            if status == "success":
                st.success("Problem solved successfully!")
                display_solution(result)

            elif status == "hitl_pending":
                st.warning("🔔 Human-in-the-Loop Review Required")

                st.write("**Clarification needed:**")
                for q in result.get("clarification_questions", []):
                    st.write(f"• {q}")

                st.info("Please revise the problem statement and try again.")

                if "trace" in result:
                    display_trace(result["trace"])

            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                if "trace" in result:
                    display_trace(result["trace"])

    
    with tab_history:
        st.markdown("### Session Statistics")
        stats = st.session_state.memory_store.get_session_statistics(
            st.session_state.orchestrator.session_id
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Problems", stats.get("problems_solved",0))
        with col2:
            st.metric("Solutions", stats.get("solutions_generated",0))
        with col3:
            st.metric("Avg Confidence", f"{stats.get('average_confidence',0.0):.2%}")
        with col4:
            st.metric("HITL Triggers", stats.get("hitl_triggers",0))

if __name__ == "__main__":
    main()
