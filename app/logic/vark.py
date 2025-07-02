import sys
from pathlib import Path
import streamlit as st
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

def update_learning_style_via_model(self, user_text: str):
    """Use VAK model to infer and update VARK profile from user text only"""
    try:
        vak_model = st.session_state.multimodal_processor.pipeline.models.get("vak")
        if vak_model:
            vak_type, confidence = vak_model.predict(user_text)
            if confidence >= 0.6:  # only apply confident updates
                logger.info(f"message: {user_text}")
                logger.info(f"🔁 VARK update from USER message: {vak_type} ({confidence:.2f})")

                # Blend into existing profile using weighted average
                for key in st.session_state.vark_profile:
                    if key == vak_type:
                        st.session_state.vark_profile[key] = (
                            st.session_state.vark_profile[key] * 0.7 + confidence * 0.3
                        )
                    else:
                        st.session_state.vark_profile[key] *= 0.95  # slight decay

                # Normalize
                total = sum(st.session_state.vark_profile.values())
                for key in st.session_state.vark_profile:
                    st.session_state.vark_profile[key] /= total
    except Exception as e:
        logger.warning(f"⚠️ VARK model update failed: {e}")