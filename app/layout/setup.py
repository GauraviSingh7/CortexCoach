import streamlit as st

def render_session_setup(self):
        """Render session initialization interface"""
        
        st.header("🚀 Start Your Coaching Session")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Welcome to your AI coaching session! This system uses the GROW model 
            (Goal, Reality, Options, Will) to help you achieve your objectives.
            
            **Features:**
            - 🎯 Structured GROW model coaching
            - 😊 Emotion-aware responses
            - 🎨 Adaptive to your learning style (VARK)
            - 📊 Real-time session analytics
            """)
            
            # User input
            user_name = st.text_input("Your Name (optional)", placeholder="Enter your name")
            user_goal = st.text_area(
                "What would you like to work on today?", 
                placeholder="Describe your goal or challenge...",
                height=100
            )
            
            # Learning style preference
            if st.button("Start Coaching Session", type="primary", disabled=not user_goal.strip()):
                self.start_session(user_name, user_goal)
        
        with col2:
            st.image("https://www.shutterstock.com/shutterstock/photos/1256570848/display_1500/stock-photo-serious-professional-female-advisor-consulting-client-at-meeting-talking-having-business-1256570848.jpg", 
                    caption="Your AI Coaching Assistant")