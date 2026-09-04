import streamlit as st
import uuid

from rag import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Corvit AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LIGHT THEME + ATTRACTIVE BACKGROUND
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION BACKGROUND
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(14, 165, 233, 0.15),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef5ff 45%,
                #f8f5ff 100%
            ) !important;

        min-height: 100vh;
    }


    /* ========================================================
       TOP STREAMLIT HEADER
       Removes the WHITE area above Corvit AI Assistant
       ======================================================== */

    header[data-testid="stHeader"] {
        background:
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef5ff 45%,
                #f8f5ff 100%
            ) !important;

        border-bottom: 1px solid rgba(219, 229, 245, 0.7) !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }


    /* ========================================================
       MAIN APP CONTAINERS
       ======================================================== */

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }


    /* ========================================================
       BOTTOM CHAT INPUT AREA
       Removes WHITE background behind the prompt box
       ======================================================== */

    [data-testid="stBottomBlockContainer"] {
        background:
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef5ff 45%,
                #f8f5ff 100%
            ) !important;
    }

    [data-testid="stBottomBlockContainer"] > div {
        background: transparent !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }


    /* ========================================================
       MAIN CONTENT WIDTH
       ======================================================== */

    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 6rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #eef5ff 0%,
                #f6f2ff 100%
            ) !important;

        border-right: 1px solid #dbe5f5;
    }


    /* ========================================================
       SIDEBAR TITLE
       ======================================================== */

    [data-testid="stSidebar"] h1 {
        color: #273c75;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    h1 {
        color: #273c75 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }


    /* ========================================================
       SUBTITLE
       ======================================================== */

    .subtitle {
        color: #64748b !important;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: 10px;
    }


    /* ========================================================
       USER MESSAGE
       ======================================================== */

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        background: rgba(219, 234, 254, 0.65);
    }


    /* ========================================================
       ASSISTANT MESSAGE
       ======================================================== */

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"]
    ) {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #e2e8f0;
    }


    /* ========================================================
       SUGGESTION BUTTONS
       ======================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 58px;

        border-radius: 14px;

        border: 1px solid #cbd5e1;

        background: rgba(255, 255, 255, 0.85);

        color: #334155;

        font-size: 14px;
        font-weight: 600;

        transition:
            all 0.2s ease;
    }


    div.stButton > button:hover {
        border-color: #6366f1;

        color: #4338ca;

        background: #ffffff;

        transform: translateY(-2px);

        box-shadow:
            0 8px 20px
            rgba(79, 70, 229, 0.12);
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #dbe5f5;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    first_chat_id = str(uuid.uuid4())

    st.session_state.chats = {
        first_chat_id: {
            "title": "New Chat",
            "messages": []
        }
    }

    st.session_state.current_chat = first_chat_id


# ============================================================
# GET CURRENT CHAT
# ============================================================

current_chat = st.session_state.chats[
    st.session_state.current_chat
]

messages = current_chat["messages"]


# ============================================================
# SIDEBAR
# ONLY CHAT HISTORY
# ============================================================

with st.sidebar:

    st.title("💬 Chat History")

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        new_chat_id = str(uuid.uuid4())

        st.session_state.chats[new_chat_id] = {
            "title": "New Chat",
            "messages": []
        }

        st.session_state.current_chat = new_chat_id

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    chat_items = list(
        st.session_state.chats.items()
    )

    chat_items.reverse()


    for chat_id, chat_data in chat_items:

        title = chat_data["title"]

        if chat_id == st.session_state.current_chat:

            button_text = f"🟢 {title}"

        else:

            button_text = f"💬 {title}"


        if st.button(
            button_text,
            key=f"chat_history_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat = chat_id

            st.rerun()


    st.divider()


    # --------------------------------------------------------
    # CLEAR CURRENT CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Current Chat",
        use_container_width=True
    ):

        st.session_state.chats[
            st.session_state.current_chat
        ]["messages"] = []

        st.session_state.chats[
            st.session_state.current_chat
        ]["title"] = "New Chat"

        st.rerun()


# ============================================================
# MAIN HEADER
# NATIVE STREAMLIT - NO HTML
# ============================================================

st.title(
    "🤖 Corvit AI Assistant"
)

st.markdown(
    "Ask anything about Corvit Systems, courses, campuses, fees and training programs."
)

st.divider()


# ============================================================
# SUGGESTION QUESTIONS
# ============================================================

selected_question = None


if len(messages) == 0:

    st.markdown(
        "### 👋 How can I help you?"
    )

    st.caption(
        "Choose a question below or type your own question."
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # SUGGESTION 1
    # --------------------------------------------------------

    with col1:

        if st.button(
            "🎓 What AI courses does Corvit offer?",
            use_container_width=True,
            key="suggestion_ai"
        ):

            selected_question = (
                "What AI courses does Corvit offer?"
            )


    # --------------------------------------------------------
    # SUGGESTION 2
    # --------------------------------------------------------

    with col2:

        if st.button(
            "📍 Where is the Rawalpindi campus?",
            use_container_width=True,
            key="suggestion_rawalpindi"
        ):

            selected_question = (
                "Where is the Rawalpindi campus?"
            )


    # --------------------------------------------------------
    # SUGGESTION 3
    # --------------------------------------------------------

    with col3:

        if st.button(
            "💰 How much is AI Machine Learning?",
            use_container_width=True,
            key="suggestion_fee"
        ):

            selected_question = (
                "How much is AI Machine Learning?"
            )


    st.write("")
    st.write("")


# ============================================================
# DISPLAY EXISTING CHAT HISTORY
# ============================================================

for message in messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# NORMAL CHAT INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask something about Corvit..."
)


if typed_question:

    selected_question = typed_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if selected_question:

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": selected_question
        }
    )


    # --------------------------------------------------------
    # CREATE CHAT TITLE
    # --------------------------------------------------------

    if current_chat["title"] == "New Chat":

        if len(selected_question) > 35:

            current_chat["title"] = (
                selected_question[:35] + "..."
            )

        else:

            current_chat["title"] = selected_question


    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            selected_question
        )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                answer, _ = generate_answer(
                    selected_question,
                    messages[:-1]
                )


                # --------------------------------------------
                # DISPLAY ONLY ANSWER
                # --------------------------------------------

                st.markdown(answer)


                # --------------------------------------------
                # SAVE ANSWER
                # --------------------------------------------

                messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                print(
                    "RAG ERROR:",
                    str(e)
                )


                error_message = (
                    "Sorry, I couldn't process your "
                    "question right now. Please try again."
                )


                st.error(
                    error_message
                )


                messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )


    # --------------------------------------------------------
    # REFRESH PAGE
    # --------------------------------------------------------

    st.rerun()