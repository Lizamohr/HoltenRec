
import os
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Village of Holton Recreation Assistant",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Village of Holton Recreation Assistant")
st.caption("A Streamlit Community Cloud demo using OpenAI + tools + a small PDF knowledge base.")


# -------------------------
# Config / paths
# -------------------------
BASE_DIR = Path(__file__).parent
PLANS_PATH = BASE_DIR / "membership_plans.csv"
FEES_PATH = BASE_DIR / "fee_components.csv"
DATA_DIR = BASE_DIR / "data_holton_rec"
PERSIST_DIR = BASE_DIR / ".chroma_holton"


# -------------------------
# API key
# -------------------------
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Community Cloud under App Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_tables():
    plans = pd.read_csv(PLANS_PATH)
    fees = pd.read_csv(FEES_PATH)
    return plans, fees


PLANS, FEES = load_tables()


# -------------------------
# RAG setup
# -------------------------
@st.cache_resource(show_spinner=True)
def build_retriever():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing knowledge-base folder: {DATA_DIR}")

    docs = DirectoryLoader(str(DATA_DIR), loader_cls=PyPDFLoader).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    vectordb = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


# -------------------------
# Helper functions
# -------------------------
def format_docs(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = os.path.basename(meta.get("source", ""))
        page = meta.get("page", "")
        lines.append(f"[{i}] {src} p{page}\n{d.page_content}")
    return "\n\n".join(lines)


# -------------------------
# Deterministic tools
# -------------------------
def lookup_plan(program: str, season: str, resident_status: str, tier: str, duration: str):
    program = program.lower().strip()
    season = season.lower().strip()
    resident_status = resident_status.lower().strip()
    tier = tier.lower().strip()
    duration = duration.lower().strip()

    if season not in ["summer", "any"]:
        season = "any"

    matches = PLANS[
        (PLANS["program"] == program) &
        (PLANS["resident_status"] == resident_status) &
        (PLANS["tier"] == tier) &
        (PLANS["duration"] == duration) &
        ((PLANS["season"] == season) | (PLANS["season"] == "any"))
    ]

    if matches.empty:
        return {
            "eligible": False,
            "reason": "No matching plan for that program/tier/duration/residency/season in this demo.",
            "plan_id": None
        }

    row = matches.iloc[0].to_dict()
    return {"eligible": True, "reason": None, **row}


def compute_invoice(plan_id: str, guest_visits: int, discount_percent: float | None = None, guest_day_pass: float | None = 8.0):
    plan_id = plan_id.strip()
    discount_percent = float(discount_percent) if discount_percent is not None else 0.0
    guest_day_pass = float(guest_day_pass) if guest_day_pass is not None else 8.0
    guest_visits = max(0, int(guest_visits))

    line_items = FEES[FEES["plan_id"] == plan_id].copy()
    if line_items.empty:
        return {"ok": False, "reason": f"No fee components found for plan_id={plan_id}"}

    membership_subtotal = float(line_items["amount"].sum())
    discount_amount = membership_subtotal * (discount_percent / 100.0)
    membership_total = membership_subtotal - discount_amount

    guest_fees = guest_visits * float(guest_day_pass)
    grand_total = membership_total + guest_fees

    return {
        "ok": True,
        "plan_id": plan_id,
        "line_items": line_items[["fee_type", "amount", "refundable"]].to_dict(orient="records"),
        "membership_subtotal": round(membership_subtotal, 2),
        "discount_percent": discount_percent,
        "discount_amount": round(discount_amount, 2),
        "membership_total": round(membership_total, 2),
        "guest_visits": guest_visits,
        "guest_day_pass": float(guest_day_pass),
        "guest_fees": round(guest_fees, 2),
        "grand_total": round(grand_total, 2),
    }


def check_refund_eligibility(purchase_date: str, membership_type: str):
    membership_type = membership_type.lower().strip()
    p = date.fromisoformat(purchase_date)
    today = date.today()
    days = (today - p).days

    if days < 0:
        return {"ok": False, "refundable": None, "reason": "Purchase date is in the future."}

    if membership_type in ["summer", "seasonal"]:
        if days <= 10:
            return {"ok": True, "refundable": True, "reason": f"Within 10-day refund window (day {days})."}
        return {"ok": True, "refundable": False, "reason": f"Outside 10-day refund window (day {days})."}

    if membership_type == "monthly":
        return {"ok": True, "refundable": False, "reason": "Monthly memberships can be canceled; access continues through end of paid month."}

    return {"ok": False, "refundable": None, "reason": f"Unknown membership_type '{membership_type}'."}


TOOL_IMPL = {
    "lookup_plan": lookup_plan,
    "compute_invoice": compute_invoice,
    "check_refund_eligibility": check_refund_eligibility,
}

TOOLS = [
    {
        "type": "function",
        "name": "lookup_plan",
        "description": "Select the correct membership plan based on program, tier, duration, residency, and season.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "tennis | fitness"},
                "season": {"type": "string", "description": "summer | any"},
                "resident_status": {"type": "string", "description": "resident | nonresident"},
                "tier": {"type": "string", "description": "individual | household | senior | premium"},
                "duration": {"type": "string", "description": "seasonal | monthly"},
            },
            "required": ["program", "season", "resident_status", "tier", "duration"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "compute_invoice",
        "description": "Compute an invoice with line items, discount, and guest day-pass fees.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "discount_percent": {"type": ["number", "null"]},
                "guest_visits": {"type": "integer"},
                "guest_day_pass": {"type": ["number", "null"]},
            },
            "required": ["plan_id", "guest_visits"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_refund_eligibility",
        "description": "Check whether a purchase is refundable based on purchase date and membership type.",
        "parameters": {
            "type": "object",
            "properties": {
                "purchase_date": {"type": "string", "description": "YYYY-MM-DD"},
                "membership_type": {"type": "string", "description": "summer | seasonal | monthly"},
            },
            "required": ["purchase_date", "membership_type"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = '''
You are the Village of Holton Recreation Assistant.
Be helpful, concise, and policy-consistent.
Use the retrieved policy excerpts for guest/refund/eligibility rules.
Use tools for pricing, calculations, and refund-window date logic.
If required information is missing (resident vs nonresident, program, season, guest visits, tier, duration, purchase timing),
ask targeted follow-up questions instead of guessing.
'''


def tool_outputs_from_response(response):
    tool_calls = [item for item in response.output if item.type == "function_call"]
    outputs = []
    executed = []

    for tc in tool_calls:
        name = tc.name
        args = json.loads(tc.arguments)

        if name == "compute_invoice" and args.get("guest_day_pass") is None:
            args["guest_day_pass"] = 8.0

        fn = TOOL_IMPL.get(name)
        result = fn(**args) if fn else {"error": f"Unknown tool {name}"}

        executed.append({"name": name, "args": args, "result": result})
        outputs.append({
            "type": "function_call_output",
            "call_id": tc.call_id,
            "output": json.dumps(result)
        })

    return outputs, executed


def run_assistant(user_text: str):
    retrieved_docs = retriever.invoke(user_text)
    context = format_docs(retrieved_docs)
    user_with_context = f"{user_text}\n\nRetrieved policy context:\n{context}"

    response = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": user_with_context}],
        temperature=0,
        tools=TOOLS,
        tool_choice="auto",
        instructions=SYSTEM,
    )

    initial_text = response.output_text or ""
    tool_outputs, executed = tool_outputs_from_response(response)
    final_text = initial_text

    if tool_outputs:
        followup = client.responses.create(
            model="gpt-4.1-mini",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
            instructions=(
                SYSTEM
                + " You just received tool outputs. "
                  "Combine tool outputs with the retrieved policy context to produce a final, user-friendly answer. "
                  "If more info is needed, ask the next best follow-up question."
            ),
            tool_choice="none",
        )
        final_text = followup.output_text or final_text

    return final_text, context, executed


# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "last_tools" not in st.session_state:
    st.session_state.last_tools = []


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("Demo notes")
    st.write(
        "This demo answers membership questions by combining:"
    )
    st.markdown(
        "- policy retrieval from PDFs\n"
        "- deterministic pricing/refund tools\n"
        "- OpenAI Responses API"
    )

    st.subheader("Suggested prompts")
    st.markdown(
        "- What’s the cheapest way for a nonresident to use the tennis facilities this summer?\n"
        "- Can I bring guests with a household fitness plan?\n"
        "- I bought a summer membership 8 days ago. Can I still get a refund?"
    )

    show_debug = st.checkbox("Show retrieved context and tool trace", value=False)


# -------------------------
# Chat UI
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about memberships, pricing, guests, or refunds...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, context, executed = run_assistant(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_context = context
        st.session_state.last_tools = executed


if show_debug and (st.session_state.last_context or st.session_state.last_tools):
    st.divider()
    st.subheader("Debug view")

    if st.session_state.last_context:
        with st.expander("Retrieved policy context", expanded=False):
            st.text(st.session_state.last_context)

    if st.session_state.last_tools:
        with st.expander("Tool trace", expanded=False):
            st.json(st.session_state.last_tools)
