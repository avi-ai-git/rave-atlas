"""
Rave Atlas — Streamlit entry point.

Thin shell: main() routes to three tab views.
All agent logic lives in agent.py; UI helpers are small named functions here.
Full UI is wired in Phase 12.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Rave Atlas",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Rave Atlas")
    st.caption("Berlin's electronic music scene — weekend planner, music education, and set-list builder.")

    tab_weekend, tab_learn, tab_crate = st.tabs(
        ["🗓 This Weekend", "📚 Learn", "🎛 Crate"]
    )

    with tab_weekend:
        st.info("Weekend concierge — coming in Phase 12.")

    with tab_learn:
        st.info("Music education chat — coming in Phase 12.")

    with tab_crate:
        st.info("Set-list builder — coming in Phase 12.")


if __name__ == "__main__":
    main()
