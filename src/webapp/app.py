import streamlit as st
import sys
from pathlib import Path
import logging
from typing import Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.rag_system import MedicalRAGSystem

class MedicalQAApp:
    """Medical QA Web Application using RAG and MLX"""
    
    def __init__(self):
        self.setup_logging()
        self.initialize_components()
        self.setup_session_state()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize RAG system"""
        try:
            self.rag_system = MedicalRAGSystem()
            self.logger.info("Successfully initialized RAG system")
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            st.error("Failed to initialize application components, please check the logs.")
            raise
            
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
            
    def format_response_display(self, response: Dict[str, Any]) -> None:
        """Format and display the response with context"""
        try:
            # Display main response with proper formatting
            st.markdown("### Response")
            if isinstance(response, dict):
                response_text = response.get('response', '')
                st.markdown(response_text)
                
                # Display confidence
                confidence = response.get('confidence', 0.0)
                if confidence > 0:
                    confidence_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
                    st.markdown(f"""
                        <div style='color: {confidence_color}'>
                            Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                # Display contexts
                contexts = response.get('contexts', [])
                if contexts:
                    with st.expander("📚 View Source Information", expanded=False):
                        st.markdown("### Reference Information")
                        for idx, ctx in enumerate(contexts, 1):
                            similarity = ctx.get('similarity', 0)
                            relevance_color = 'green' if similarity > 0.7 else 'orange' if similarity > 0.4 else 'red'
                            
                            st.markdown(f"""
                            <div style='padding: 10px; border-left: 3px solid {relevance_color}; margin: 10px 0;'>
                                <strong>Source {idx}</strong> (Relevance: {similarity:.1%})
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Q:** {ctx['question']}")
                            st.markdown(f"**A:** {ctx['answer']}")
                            st.markdown("---")
            else:
                st.markdown(str(response))
                
        except Exception as e:
            st.error(f"Error displaying response: {str(e)}")

    def handle_user_input(self, user_input: str):
        """Process user input and generate response"""
        try:
            # Add user message
            st.session_state.conversation.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response
            with st.spinner("🔍 Searching medical knowledge base..."):
                response = self.rag_system.generate_response(user_input)
                
            # Add assistant message
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response
            })
            
            # Format and display response
            self.format_response_display(response)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            st.error("An error occurred while processing your question, please try again.")

    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Medical Bot Assistant",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 Medical Question-Answering System")
        
        # Add information about the system
        st.markdown("""
        Welcome! This AI-powered medical assistant can help answer your health-related questions 
        using a comprehensive medical knowledge base. 
        
        **Features:**
        - Access to verified medical Q&A dataset
        - Semantic search for relevant information
        - Source references with relevance scores for transparency
        
        ⚠️ **Important Note:** This system provides information for educational purposes only 
        and should not replace professional medical advice.
        """)
        
        # User input form
        with st.form(key="question_form"):
            user_input = st.text_area(
                "💭 Type your medical question:",
                height=100,
                placeholder="Example: What are the symptoms of diabetes?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button(
                    "🔍 Ask Question",
                    use_container_width=True
                )
                
        if submit_button and user_input:
            self.handle_user_input(user_input)
        
        # Display conversation history
        if st.session_state.conversation:
            st.markdown("### 💬 Conversation History")
            for message in st.session_state.conversation:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(f"👤 **You:** {content}")
                else:
                    st.markdown("🤖 **Assistant:**")
                    self.format_response_display(content)
                    st.markdown("---")
        
        # Footer with disclaimer
        st.markdown("""
        ---
        **📋 Medical Disclaimer:** This AI assistant provides general information only. 
        Always consult qualified healthcare professionals for medical advice and treatment decisions.
        """)

if __name__ == "__main__":
    app = MedicalQAApp()
    app.run()