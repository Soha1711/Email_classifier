import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

API_URL = "http://16.171.176.15:8000/predict"

st.set_page_config(page_title="AI Email Classifier", layout="wide")

# ===== LABEL MAPPING FIX =====
CATEGORY_MAP = {
    "0": "complaint",
    "1": "request",
    "2": "feedback",
    "3": "spam"
}

URGENCY_MAP = {
    "0": "low",
    "1": "medium",
    "2": "high"
}


# ===== BADGE FORMAT =====
def badge(cat, urg):

    # CATEGORY BADGE
    if cat == "complaint":
        cat_badge = f"🟥 {cat}"
    elif cat == "request":
        cat_badge = f"🟦 {cat}"
    elif cat == "feedback":
        cat_badge = f"🟩 {cat}"
    else:
        cat_badge = f"⬛ {cat}"

    # URGENCY BADGE
    if urg == "high":
        urg_badge = f"🔴 {urg}"
    elif urg == "medium":
        urg_badge = f"🟡 {urg}"
    else:
        urg_badge = f"🟢 {urg}"

    return cat_badge, urg_badge


# ===== CORPORATE INFOSYS STYLE =====
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #f0f6ff, #ffffff);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B5ED7, #084298) !important;
    color: white !important;
}

.title {
    font-size: 26px;
    font-weight: 700;
    color: #0B5ED7;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #dbeafe;
}
</style>
""", unsafe_allow_html=True)


# ===== SESSION STATE =====
if "login" not in st.session_state:
    st.session_state.login = False

if "emails" not in st.session_state:
    st.session_state.emails = []


# ===== BACKEND CALL =====
def classify(sender, subject, text):
    try:
        r = requests.post(API_URL, json={
            "sender": sender,
            "subject": subject,
            "text": text
        })

        data = r.json()

        # ----- FIX START -----
        cat = data.get("category")
        urg = data.get("urgency")

        # Convert STRING → INT safely
        try:
            cat = int(cat)
            urg = int(urg)
        except:
            pass

        # Now map to names
        cat = CATEGORY_MAP.get(cat, "request")
        urg = URGENCY_MAP.get(urg, "medium")
        # ----- FIX END -----

        data["category"] = cat
        data["urgency"] = urg

        return data

    except Exception as e:
        st.error(f"Backend Error: {e}")
        return None



# ===== LOGIN PAGE =====
def login():
    c1, c2, c3 = st.columns([1,1,1])

    with c2:
       # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>       User Login     </div>", unsafe_allow_html=True)

        u = st.text_input("Email")
        p = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if u=="soha7@gmail.com" and p=="soha1711":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.markdown("</div>", unsafe_allow_html=True)


# ===== MAIN DASHBOARD =====
def main():

    st.sidebar.markdown("## AI EMAIL CLASSIFIER")

    menu = st.sidebar.radio("Navigation", [
        "Dashboard",
        "Classify Email",
        "Inbox"
    ])

    # ===== DASHBOARD =====
    if menu == "Dashboard":

        st.markdown("<div class='title'>Email Intelligence Dashboard</div>", unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.emails)

        if not df.empty:

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Emails", len(df))
            c2.metric("Top Category", df["category"].mode()[0])
            c3.metric("Most Urgent", df["urgency"].mode()[0])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(df, x="category", title="Category Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.pie(df, names="urgency", title="Urgency Levels")
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No emails classified yet")


    # ===== CLASSIFY EMAIL =====
    if menu == "Classify Email":

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>AI Email Classification</div>", unsafe_allow_html=True)

        sender = st.text_input("Sender")
        subject = st.text_input("Subject")
        text = st.text_area("Email Content", height=180)

        if st.button("Analyze Email"):

            res = classify(sender,subject,text)

            if res:

                cat = res.get("category")
                urg = res.get("urgency")

                cat_badge, urg_badge = badge(cat, urg)

                new = {
                    "sender": sender,
                    "subject": subject,
                    "body": text,
                    "category": cat,
                    "urgency": urg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }

                st.session_state.emails.insert(0,new)

                st.success("Email Classified Successfully")

                st.info(f"Category: {cat_badge} | Urgency: {urg_badge}")

        st.markdown("</div>", unsafe_allow_html=True)


    # ===== INBOX =====
    if menu == "Inbox":

        st.markdown("<div class='title'>Live Email Feed</div>", unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.emails)

        if df.empty:
            st.info("No Emails Classified Yet")
            return

        df.index = df.index + 1

        st.dataframe(
            df[["sender","subject","category","urgency","timestamp"]],
            use_container_width=True
        )

        s = st.selectbox("Open Email", df.subject)

        mail = df[df.subject==s].iloc[0]

        cat_badge, urg_badge = badge(mail.category, mail.urgency)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.write("From:", mail.sender)
        st.write("Category:", cat_badge)
        st.write("Urgency:", urg_badge)
        st.write("Message:", mail.body)

        st.markdown("</div>", unsafe_allow_html=True)


# ===== RUN APP =====
if not st.session_state.login:
    login()
else:
    main()
