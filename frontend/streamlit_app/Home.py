
import os
import json
import requests
import streamlit as st

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND

st.set_page_config(page_title="Social Support – MVP", layout="wide")
st.title("Social Support – Local MVP (Phases 1–3)")

with st.sidebar:
    st.header("Settings")
    st.session_state.backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="FastAPI base URL (default http://127.0.0.1:8000)",
    )
    backend = st.session_state.backend_url
    if st.button("Check Health"):
        try:
            r = requests.get(f"{backend}/healthz", timeout=10)
            if r.ok:
                st.success(f"Backend healthy: {r.text}")
            else:
                st.error(f"Health check failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Health check error: {e}")

def post_files(url: str, form_data: dict, files: list):
    files_payload = []
    for f in files or []:
        if f is None:
            continue
        files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
    return requests.post(url, data=form_data, files=files_payload, timeout=120)

def show_json(label: str, data):
    st.subheader(label)
    if isinstance(data, (dict, list)):
        st.json(data)
    else:
        try:
            st.json(json.loads(str(data)))
        except Exception:
            st.code(str(data))

# Phase 1 — Ingest
st.header("Apply (Phase 1)")
with st.form("ingest_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    with col1:
        name = st.text_input("Full Name", "")
        emirates_id = st.text_input("Emirates ID", "")
    with col2:
        declared_income = st.number_input("Declared Monthly Income (AED)", min_value=0.0, value=0.0, step=100.0)
    with col3:
        household_size = st.number_input("Household Size", min_value=1, value=1, step=1)
    with col4:
        channel = st.selectbox("Channel", ["Online", "In-Person", "Assisted"], index=0)

    files = st.file_uploader(
        "Upload supporting documents (PDF/JPG/PNG/XLSX/CSV)",
        type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Submit Application")

if submitted:
    try:
        payload = {
            "name": name,
            "emirates_id": emirates_id,
            "declared_monthly_income": str(declared_income),
            "household_size": str(household_size),
            "channel": channel,
        }
        r = post_files(f"{backend}/ingest", form_data=payload, files=files)
        if r.ok:
            resp = r.json()
            st.success("Application submitted.")
            show_json("Response", resp)
            app_id = resp.get("application_id")
            if app_id:
                st.session_state.last_application_id = app_id
                st.info(f"Application ID: {app_id}")
        else:
            st.error(f"Failed to submit: {r.status_code}")
            st.code(r.text)
    except Exception as e:
        st.error(f"Ingest error: {e}")

# Phase 2 — Extraction
st.divider()
st.header("Extraction (Phase 2)")
default_extract_id = st.session_state.get("last_application_id", "")
app_id_for_extract = st.text_input("Application ID to extract", value=default_extract_id, key="extract_id")

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Run Extraction", disabled=not app_id_for_extract.strip()):
        try:
            r = requests.post(f"{backend}/extract/{app_id_for_extract.strip()}", timeout=600)
            if r.ok:
                resp = r.json()
                st.success("Extraction completed.")
                if isinstance(resp, dict):
                    if "summary" in resp: show_json("Summary", resp["summary"])
                    if "ocr_reports" in resp: show_json("OCR Reports", resp["ocr_reports"])
                    if "parse_reports" in resp: show_json("Parse Reports", resp["parse_reports"])
                    if not any(k in resp for k in ("summary","ocr_reports","parse_reports")):
                        show_json("Extraction Response", resp)
                else:
                    show_json("Extraction Response", resp)
            else:
                st.error(f"Extraction failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Extraction error: {e}")

with colB:
    if st.button("Open Extraction Folder", disabled=not app_id_for_extract.strip()):
        st.info(
            "Artifacts are saved under:\n"
            f"`data/apps/{app_id_for_extract.strip()}/`  \n"
            "Files: `ocr.json`, `parsed.json`, `extraction_summary.json`"
        )

# Phase 3 — Validation
st.divider()
st.header("Validation (Phase 3)")
default_validate_id = st.session_state.get("last_application_id", "")
app_id_for_validate = st.text_input("Application ID to validate", value=default_validate_id, key="validate_id")

if st.button("Run Validation", disabled=not app_id_for_validate.strip()):
    try:
        r = requests.post(f"{backend}/validate/{app_id_for_validate.strip()}", timeout=300)
        if r.ok and r.json().get("ok"):
            st.success("Validation completed.")
            st.json(r.json().get("validation", {}))
        else:
            st.error(f"Validation failed: {r.status_code}")
            st.code(r.text)
    except Exception as e:
        st.error(f"Validation error: {e}")

# Phase 3 — Orchestrator
st.divider()
st.header("Process Application (Phase 3)")
default_proc_id = st.session_state.get("last_application_id", "")
app_id_proc = st.text_input("Application ID to process", value=default_proc_id, key="proc_id")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Run Orchestrator", disabled=not app_id_proc.strip()):
        try:
            r = requests.post(f"{backend}/process/{app_id_proc.strip()}", timeout=900)
            if r.ok and r.json().get("ok"):
                st.success("Orchestration completed.")
                out = r.json().get("result", {})
                st.subheader("Final State"); st.json(out.get("state", {}))
                st.subheader("Trace"); st.json(out.get("trace", []))
            else:
                st.error(f"Process failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Process error: {e}")

with col2:
    if st.button("Load Last Trace", disabled=not app_id_proc.strip()):
        try:
            r = requests.get(f"{backend}/process/{app_id_proc.strip()}/trace", timeout=60)
            if r.ok:
                st.success("Loaded saved trace."); st.json(r.json())
            else:
                st.error(f"Trace fetch failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Trace error: {e}")

# Phase 4 — ML Scoring
st.divider()
st.header("ML Scoring (Phase 4)")
default_score_id = st.session_state.get("last_application_id", "")
app_id_score = st.text_input("Application ID to score", value=default_score_id, key="score_id")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Run ML Scoring", disabled=not app_id_score.strip()):
        try:
            r = requests.post(f"{backend}/score/{app_id_score.strip()}", timeout=300)
            if r.ok and r.json().get("ok"):
                st.success("ML Scoring completed.")
                ml_result = r.json().get("ml_result", {})

                # Decision summary
                st.subheader("ML Decision")
                decision = ml_result.get("decision", "Unknown")
                confidence = ml_result.get("confidence", 0)

                if decision == "APPROVE":
                    st.success(f"✅ **{decision}** (Confidence: {confidence:.1%})")
                else:
                    st.error(f"❌ **{decision}** (Confidence: {confidence:.1%})")

                # Key factors
                st.subheader("Key Decision Factors")
                top_factors = ml_result.get("top_factors", [])
                for i, factor in enumerate(top_factors[:3], 1):
                    impact_color = "🔼" if factor["impact"] > 0 else "🔽"
                    st.write(f"{i}. {impact_color} **{factor['feature']}**: {factor['value']:.1f} "
                            f"({factor['direction']} impact: {factor['impact']:.3f})")

                # Full SHAP explanation
                with st.expander("Full SHAP Explanation"):
                    st.json(ml_result.get("shap_explanation", {}))

                # Probabilities
                with st.expander("Prediction Probabilities"):
                    approve_prob = ml_result.get("approve_probability", 0)
                    decline_prob = ml_result.get("decline_probability", 0)
                    st.write(f"Approve: {approve_prob:.1%}")
                    st.write(f"Decline: {decline_prob:.1%}")

            else:
                st.error(f"ML Scoring failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"ML Scoring error: {e}")

with col2:
    if st.button("Load Saved ML Score", disabled=not app_id_score.strip()):
        try:
            r = requests.get(f"{backend}/score/{app_id_score.strip()}", timeout=60)
            if r.ok and r.json().get("ok"):
                st.success("Loaded saved ML score.")
                ml_result = r.json().get("ml_result", {})

                # Quick summary
                decision = ml_result.get("decision", "Unknown")
                confidence = ml_result.get("confidence", 0)
                st.info(f"Decision: **{decision}** (Confidence: {confidence:.1%})")

                # Show top factors
                top_factors = ml_result.get("top_factors", [])
                if top_factors:
                    st.write("**Top Decision Factors:**")
                    for factor in top_factors[:2]:
                        impact_icon = "🔼" if factor["impact"] > 0 else "🔽"
                        st.write(f"- {impact_icon} {factor['feature']}: {factor['value']:.1f}")

            else:
                st.error(f"Failed to load ML score: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Load ML score error: {e}")

# Phase 5 — Graph Analysis & Duplicate Detection
st.divider()
st.header("Graph Analysis & Duplicate Detection (Phase 5)")
default_graph_id = st.session_state.get("last_application_id", "")
app_id_graph = st.text_input("Application ID for graph analysis", value=default_graph_id, key="graph_id")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Run Graph Analysis", disabled=not app_id_graph.strip()):
        try:
            r = requests.post(f"{backend}/graph/analyze/{app_id_graph.strip()}", timeout=600)
            if r.ok and r.json().get("ok"):
                st.success("Graph Analysis completed.")
                result = r.json()

                # Summary stats
                st.subheader("Entity Extraction Summary")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Entities Extracted", result.get("entities_extracted", 0))
                with col_b:
                    st.metric("Entities Deduplicated", result.get("entities_deduplicated", 0))
                with col_c:
                    graph_results = result.get("graph_results", {})
                    nodes_created = graph_results.get("nodes_created", 0)
                    st.metric("Graph Nodes Created", nodes_created)

                # Show warnings if any
                if "warning" in graph_results:
                    st.warning(f"Graph Warning: {graph_results['warning']}")

            else:
                st.error(f"Graph analysis failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Graph analysis error: {e}")

with col2:
    if st.button("Load Graph Analysis", disabled=not app_id_graph.strip()):
        try:
            r = requests.get(f"{backend}/graph/analysis/{app_id_graph.strip()}", timeout=60)
            if r.ok and r.json().get("ok"):
                st.success("Loaded graph analysis.")
                analysis = r.json().get("analysis", {})

                # Entity counts
                entities = analysis.get("entities", [])
                entity_types = {}
                for entity in entities:
                    etype = entity.get("type", "unknown")
                    entity_types[etype] = entity_types.get(etype, 0) + 1

                st.subheader("Extracted Entities by Type")
                for etype, count in entity_types.items():
                    st.write(f"- **{etype.title()}**: {count}")

            else:
                st.error(f"Failed to load graph analysis: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Load graph analysis error: {e}")

# Duplicate Detection Section
st.subheader("Duplicate Detection")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Detect Duplicates", disabled=not app_id_graph.strip()):
        try:
            r = requests.post(f"{backend}/duplicates/detect/{app_id_graph.strip()}", timeout=300)
            if r.ok and r.json().get("ok"):
                results = r.json().get("results", {})
                duplicates_found = results.get("duplicates_found", 0)

                if duplicates_found > 0:
                    st.error(f"🚨 **{duplicates_found} Duplicate(s) Detected!**")

                    # Show duplicate statistics
                    stats = results.get("statistics", {})
                    if stats:
                        st.subheader("Risk Distribution")
                        risk_dist = stats.get("risk_distribution", {})
                        col_high, col_med, col_low = st.columns(3)
                        with col_high:
                            st.metric("High Risk", risk_dist.get("high", 0), delta=None)
                        with col_med:
                            st.metric("Medium Risk", risk_dist.get("medium", 0), delta=None)
                        with col_low:
                            st.metric("Low Risk", risk_dist.get("low", 0), delta=None)

                    # Show duplicate details
                    duplicates = results.get("duplicates", [])
                    for i, dup in enumerate(duplicates, 1):
                        with st.expander(f"Duplicate {i}: {dup['duplicate_application_id']} (Risk: {dup['risk_level'].upper()})"):
                            st.write(f"**Match Type**: {dup['match_type']}")
                            st.write(f"**Similarity Score**: {dup['similarity_score']:.3f}")
                            st.write(f"**Confidence**: {dup['confidence']:.3f}")

                            st.subheader("Evidence")
                            for evidence in dup.get("evidence", []):
                                st.write(f"- **{evidence['type']}**: {evidence.get('similarity', 'N/A'):.3f} similarity")
                                if 'current' in evidence and 'historical' in evidence:
                                    st.write(f"  - Current: {evidence['current']}")
                                    st.write(f"  - Historical: {evidence['historical']}")
                else:
                    st.success("✅ No duplicates detected")

            else:
                st.error(f"Duplicate detection failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Duplicate detection error: {e}")

with col2:
    if st.button("Load Duplicates", disabled=not app_id_graph.strip()):
        try:
            r = requests.get(f"{backend}/duplicates/{app_id_graph.strip()}", timeout=60)
            if r.ok and r.json().get("ok"):
                results = r.json().get("results", {})
                duplicates_found = results.get("duplicates_found", 0)

                if duplicates_found > 0:
                    st.warning(f"⚠️ {duplicates_found} duplicate(s) found")
                    stats = results.get("statistics", {})
                    if stats.get("highest_risk"):
                        st.write(f"Highest Risk Level: **{stats['highest_risk'].upper()}**")
                else:
                    st.info("No duplicates in saved analysis")

            else:
                st.error(f"Failed to load duplicates: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Load duplicates error: {e}")

# Conflict Detection Section
st.subheader("Conflict Detection")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Detect Conflicts", disabled=not app_id_graph.strip()):
        try:
            r = requests.post(f"{backend}/conflicts/detect/{app_id_graph.strip()}", timeout=300)
            if r.ok and r.json().get("ok"):
                results = r.json().get("results", {})
                conflicts_found = results.get("conflicts_found", 0)

                if conflicts_found > 0:
                    st.warning(f"⚠️ **{conflicts_found} Conflict(s) Detected!**")

                    conflicts = results.get("conflicts", [])
                    for i, conflict in enumerate(conflicts, 1):
                        severity = conflict["severity"]
                        severity_color = "🔴" if severity == "critical" else "🟡" if severity == "warning" else "🔵"

                        with st.expander(f"Conflict {i}: {conflict['type']} ({severity_color} {severity.upper()})"):
                            st.write(f"**Description**: {conflict['description']}")

                            # Show evidence
                            evidence = conflict.get("evidence", {})
                            if evidence:
                                st.subheader("Evidence")
                                for key, value in evidence.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"- **{key}**: {value:.2f}")
                                    else:
                                        st.write(f"- **{key}**: {value}")

                            # Show suggestions
                            suggestions = conflict.get("suggestions", [])
                            if suggestions:
                                st.subheader("Suggested Actions")
                                for suggestion in suggestions:
                                    st.write(f"• {suggestion}")
                else:
                    st.success("✅ No conflicts detected")

            else:
                st.error(f"Conflict detection failed: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Conflict detection error: {e}")

with col2:
    if st.button("Load Conflicts", disabled=not app_id_graph.strip()):
        try:
            r = requests.get(f"{backend}/conflicts/{app_id_graph.strip()}", timeout=60)
            if r.ok and r.json().get("ok"):
                results = r.json().get("results", {})
                conflicts_found = results.get("conflicts_found", 0)

                if conflicts_found > 0:
                    st.warning(f"⚠️ {conflicts_found} conflict(s) found")

                    # Show severity breakdown
                    conflicts = results.get("conflicts", [])
                    severity_counts = {"critical": 0, "warning": 0, "info": 0}
                    for conflict in conflicts:
                        severity = conflict.get("severity", "info")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1

                    if severity_counts["critical"] > 0:
                        st.error(f"🔴 {severity_counts['critical']} critical")
                    if severity_counts["warning"] > 0:
                        st.warning(f"🟡 {severity_counts['warning']} warnings")
                    if severity_counts["info"] > 0:
                        st.info(f"🔵 {severity_counts['info']} info")
                else:
                    st.info("No conflicts in saved analysis")

            else:
                st.error(f"Failed to load conflicts: {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Load conflicts error: {e}")

# Phase 6 — AI Chat Application
st.divider()
st.header("AI Chat Application (Phase 6) 💬")

# Initialize session state for chat
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_processing" not in st.session_state:
    st.session_state.chat_processing = False
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0

def start_new_chat():
    try:
        r = requests.post(f"{backend}/chat/session", timeout=30)
        if r.ok and r.json().get("ok"):
            st.session_state.chat_session_id = r.json()["session_id"]
            st.session_state.chat_messages = []
            st.session_state.message_counter = 0  # Reset counter for new session
            load_chat_messages()
            st.success("✅ New chat session started!")
            st.rerun()  # Refresh to show messages
        else:
            st.error(f"Failed to start chat: {r.status_code}")
    except Exception as e:
        st.error(f"Chat error: {e}")

def load_chat_messages():
    if not st.session_state.chat_session_id:
        return
    try:
        r = requests.get(f"{backend}/chat/{st.session_state.chat_session_id}/messages", timeout=30)
        if r.ok and r.json().get("ok"):
            st.session_state.chat_messages = r.json()["messages"]
    except Exception as e:
        st.error(f"Failed to load messages: {e}")

def send_chat_message(message):
    if not st.session_state.chat_session_id:
        st.error("No active chat session. Please start a new chat.")
        return

    if not message or not message.strip():
        return  # Don't send empty messages

    try:
        payload = {"content": message.strip(), "type": "text"}
        r = requests.post(f"{backend}/chat/{st.session_state.chat_session_id}/message", json=payload, timeout=60)
        if r.ok and r.json().get("ok"):
            # Increment counter to force input refresh
            st.session_state.message_counter += 1
            # Load messages after successful send
            load_chat_messages()
            # Force rerun to update the display
            st.rerun()
        else:
            st.error(f"Failed to send message: {r.status_code}")
    except Exception as e:
        st.error(f"Message error: {e}")

def upload_chat_file(uploaded_file):
    if not st.session_state.chat_session_id:
        st.error("No active chat session. Please start a new chat.")
        return

    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        r = requests.post(f"{backend}/chat/{st.session_state.chat_session_id}/upload", files=files, timeout=120)
        if r.ok and r.json().get("ok"):
            # Reload messages to show upload confirmation
            load_chat_messages()
            st.success(f"✅ {uploaded_file.name} uploaded successfully")
            # Force UI refresh to show updated messages
            st.rerun()
        else:
            st.error(f"Failed to upload file: {r.status_code}")
            if r.text:
                st.code(r.text)
    except Exception as e:
        st.error(f"Upload error: {e}")

def process_chat_application():
    if not st.session_state.chat_session_id:
        st.error("No active chat session.")
        return

    try:
        st.session_state.chat_processing = True
        r = requests.post(f"{backend}/chat/{st.session_state.chat_session_id}/process", timeout=300)
        if r.ok and r.json().get("ok"):
            load_chat_messages()
            result = r.json()
            st.success("✅ Application processed successfully!")

            decision = result.get("decision", "Unknown")
            confidence = result.get("confidence", 0)

            if decision == "APPROVE":
                st.success(f"🎉 **APPROVED** (Confidence: {confidence:.1%})")
            else:
                st.error(f"❌ **DECLINED** (Confidence: {confidence:.1%})")
        else:
            st.error(f"Failed to process application: {r.status_code}")
    except Exception as e:
        st.error(f"Processing error: {e}")
    finally:
        st.session_state.chat_processing = False

# Chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat with AI Assistant")

    if st.button("🆕 Start New Chat", type="primary"):
        start_new_chat()

with col2:
    if st.session_state.chat_session_id:
        st.success(f"Session: {st.session_state.chat_session_id[:8]}...")

        # Show progress status
        try:
            r = requests.get(f"{backend}/chat/{st.session_state.chat_session_id}/status", timeout=10)
            if r.ok and r.json().get("ok"):
                status = r.json()
                data_collected = status.get("data_collected", {})
                doc_count = data_collected.get("documents", 0)
                current_state = status.get("state", "unknown")

                # Show progress indicators
                if doc_count > 0:
                    st.info(f"📎 {doc_count} document(s) uploaded")

                # Show current state
                state_display = {
                    "greeting": "🏁 Getting started",
                    "collecting_name": "👤 Collecting name",
                    "collecting_emirates_id": "🆔 Collecting Emirates ID",
                    "collecting_income": "💰 Collecting income info",
                    "collecting_household_size": "🏠 Collecting household info",
                    "collecting_documents": "📋 Ready for documents",
                    "processing_application": "⚙️ Processing application",
                    "showing_results": "✅ Application processed",
                    "completed": "🎉 Complete"
                }
                state_text = state_display.get(current_state, current_state)
                st.caption(f"Status: {state_text}")
        except Exception:
            pass

        if st.button("🔄 Refresh Messages"):
            load_chat_messages()

# Display chat messages
if st.session_state.chat_session_id:
    if st.session_state.chat_messages:
        st.subheader("Conversation")

        # Display messages in a container with latest at bottom
        message_container = st.container()
        with message_container:
            for i, msg in enumerate(st.session_state.chat_messages):
                timestamp = msg.get("timestamp", "")
                sender = msg.get("sender", "")
                content = msg.get("content", "")
                msg_type = msg.get("message_type", "text")

                if sender == "user":
                    with st.chat_message("user"):
                        st.write(content)
                elif sender == "assistant":
                    with st.chat_message("assistant"):
                        if msg_type == "system":
                            st.success(content)
                        else:
                            st.write(content)

            # Add some spacing
            st.write("")
    else:
        st.subheader("Conversation")
        st.info("Loading messages...")

# Input section
if st.session_state.chat_session_id:
    st.subheader("Send Message")

    # Use form to prevent double submission on Enter
    with st.form(key=f"chat_form_{st.session_state.message_counter}", clear_on_submit=True):
        user_input = st.text_input("Type your message:", placeholder="Type your response here...", key=f"chat_input_{st.session_state.message_counter}")
        col1, col2 = st.columns([4, 1])
        with col2:
            submitted = st.form_submit_button("📤 Send", use_container_width=True)

        if submitted and user_input and user_input.strip():
            send_chat_message(user_input)

    st.write("---")  # Separator

    col1, col2 = st.columns([1, 1])

    with col1:
        # File upload
        st.subheader("📎 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose file to upload",
            type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "xlsx", "xls", "csv", "txt"],
            key="chat_file_upload",
            help="Upload your Emirates ID, bank statements, salary certificates, etc."
        )

        if uploaded_file:
            if st.button("📤 Upload File", use_container_width=True, type="primary"):
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    upload_chat_file(uploaded_file)

    with col2:
        # Process application button
        st.write("") # Spacing
        if st.button("⚡ Process Application", disabled=st.session_state.chat_processing):
            if st.session_state.chat_processing:
                st.info("Processing...")
            else:
                process_chat_application()

    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("❓ Ask about eligibility", key="quick_eligibility"):
            if st.session_state.chat_session_id:
                send_chat_message("What are the eligibility criteria for social support?")

    with col2:
        if st.button("📋 Required documents", key="quick_documents"):
            if st.session_state.chat_session_id:
                send_chat_message("What documents do I need to provide?")

    with col3:
        if st.button("⏱️ Processing time", key="quick_processing"):
            if st.session_state.chat_session_id:
                send_chat_message("How long does the application process take?")

else:
    st.info("👆 Click 'Start New Chat' to begin your AI-powered application process!")

    st.markdown("""
    ### 🤖 **AI Chat Features:**
    - **Intelligent guidance** through the application process
    - **Real-time document processing** and validation
    - **Instant feedback** on income discrepancies and issues
    - **RAG-powered responses** based on UAE social support policies
    - **Seamless integration** with existing processing pipeline
    - **Conversational experience** instead of complex forms
    """)

st.divider()
with st.expander("Tips & Notes"):
    st.markdown(
        """
- Uploaded docs -> `data/apps/<application_id>/`
- Extraction -> `ocr.json`, `parsed.json`, `extraction_summary.json`
- Validation -> `validation.json`
- ML Scoring -> `ml_score.json` (SHAP explanations and predictions)
- Graph Analysis -> `graph_analysis.json` (entities, duplicates, conflicts)
- Chat Sessions -> `data/apps/chat_<session_id>/` (documents and conversation data)
- Orchestrator -> `orchestrator_trace.json`, checkpoint in `data/orchestrator.db`
- For better OCR on macOS: `brew install tesseract ocrmypdf poppler`
- **Phase 4**: ML scoring replaces rules-based decisions with GradientBoostingClassifier + SHAP explanations
- **Phase 5**: Entity extraction, graph analysis, duplicate detection, and conflict analysis
- **Phase 6**: RAG-powered conversational application processing with Qdrant + Ollama
- For Neo4j graph storage: Install Neo4j and set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars
- For RAG functionality: Qdrant (Docker) + Ollama with nomic-embed-text model required
        """
    )
