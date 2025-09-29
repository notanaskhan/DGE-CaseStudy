
import os, uuid, json, datetime as dt
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Database imports
from backend.fastapi_app.database import (
    init_databases,
    get_current_session,
    DatabaseSession,
    Application,
    Document,
    AuditLog
)
from backend.fastapi_app.database.mongo_service import get_mongo_service

# Paths
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(REPO_DIR, "data")
APPS_DIR = os.path.join(DATA_DIR, "apps")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(APPS_DIR, exist_ok=True)

# Initialize databases
init_databases()

# Get session maker for backward compatibility
SessionLocal = get_current_session()

app = FastAPI(title="Social Support – Local MVP (Phases 1–3)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

from backend.fastapi_app.services.ocr_service import run_ocr
from backend.fastapi_app.services.parse_service import parse_document
from backend.fastapi_app.services.validation_service import validate_application
from backend.fastapi_app.agents.orchestrator import run_orchestration, _trace_path

class IngestResponse(BaseModel):
    application_id: str
    status: str

def _save_upload(app_id: str, file: UploadFile) -> str:
    dest_dir = os.path.join(APPS_DIR, app_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, file.filename)
    with open(dest_path, "wb") as out:
        out.write(file.file.read())
    return dest_path

def _audit(db, app_id: str, action: str, payload: dict):
    """Audit action to both PostgreSQL/SQLite and MongoDB."""
    # Store in primary database (PostgreSQL/SQLite)
    db.add(AuditLog(id=str(uuid.uuid4()), application_id=app_id, action=action, payload_json=json.dumps(payload)))
    db.commit()

    # Store in MongoDB for analytics (if available)
    mongo_service = get_mongo_service()
    mongo_service.store_analytics_data("audit_log", {
        "application_id": app_id,
        "action": action,
        "payload": payload
    })

@app.get("/healthz")
def healthz():
    """Health check with database status."""
    from backend.fastapi_app.database.database_config import get_database_info

    db_info = get_database_info()
    return {
        "status": "ok",
        "database": db_info
    }

@app.post("/ingest", response_model=IngestResponse)
def ingest(
    name: str = Form(""),
    emirates_id: str = Form(""),
    declared_monthly_income: float = Form(0.0),
    household_size: int = Form(1),
    channel: str = Form("Online"),
    files: List[UploadFile] = File(default_factory=list),
):
    db = SessionLocal()
    try:
        app_id = str(uuid.uuid4())
        app_row = Application(
            id=app_id,
            name=name,
            emirates_id=emirates_id,
            declared_monthly_income=declared_monthly_income,
            household_size=household_size,
            channel=channel,
            status="RECEIVED",
        )
        db.add(app_row); db.commit()
        paths = []
        for f in files or []:
            path = _save_upload(app_id, f)
            doc = Document(id=str(uuid.uuid4()), application_id=app_id, path=path, filename=os.path.basename(path))
            db.add(doc); paths.append(path)
        db.commit()
        _audit(db, app_id, "ingest", {"paths": paths})
        return {"application_id": app_id, "status": "RECEIVED"}
    finally:
        db.close()

@app.post("/extract/{application_id}")
def extract(application_id: str):
    db = SessionLocal()
    try:
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}
        docs = db.query(Document).filter(Document.application_id == application_id).all()

        dest_dir = os.path.join(APPS_DIR, application_id)
        ocr_dir = os.path.join(dest_dir, "ocr")
        os.makedirs(ocr_dir, exist_ok=True)

        ocr_reports = {}
        parse_reports = {}

        for d in docs:
            ocr_reports[d.filename] = run_ocr(d.path, ocr_dir)
            parse_reports[d.filename] = parse_document(d.path)

        with open(os.path.join(dest_dir, "ocr.json"), "w") as f:
            json.dump(ocr_reports, f, indent=2)
        with open(os.path.join(dest_dir, "parsed.json"), "w") as f:
            json.dump(parse_reports, f, indent=2)

        total_inflow = sum((r.get("total_inflow", 0.0) for r in parse_reports.values() if isinstance(r, dict)))
        total_outflow = sum((r.get("total_outflow", 0.0) for r in parse_reports.values() if isinstance(r, dict)))
        summary = {
            "application_id": application_id,
            "total_inflow": total_inflow,
            "total_outflow": total_outflow,
            "documents_processed": len(docs),
            "notes": "OCR/parsing are best-effort; missing system binaries are reported as warnings."
        }
        with open(os.path.join(dest_dir, "extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        _audit(db, application_id, "extract", {"summary": summary})
        return {"summary": summary, "ocr_reports": ocr_reports, "parse_reports": parse_reports}
    finally:
        db.close()

@app.post("/validate/{application_id}")
def validate_app(application_id: str):
    db = SessionLocal()
    try:
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}
        docs = db.query(Document).filter(Document.application_id == application_id).all()
        doc_paths = [d.path for d in docs if os.path.exists(d.path)]

        result = validate_application(
            app_row={"declared_monthly_income": app_row.declared_monthly_income, "household_size": app_row.household_size},
            doc_paths=doc_paths,
        )
        dest_dir = os.path.join(APPS_DIR, application_id)
        with open(os.path.join(dest_dir, "validation.json"), "w") as f:
            json.dump(result, f, indent=2)

        app_row.status = "VALIDATED"
        db.add(app_row); db.commit()
        _audit(db, application_id, "validate", result)
        return {"ok": True, "validation": result}
    finally:
        db.close()

@app.post("/process/{application_id}")
def process_application(application_id: str):
    try:
        result = run_orchestration(application_id)
        return {"ok": True, "result": result}
    except Exception as e:
        try:
            from backend.fastapi_app.agents.orchestrator import append_trace
            append_trace(application_id, {"step": "process", "status": "error", "error": str(e)})
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

@app.get("/process/{application_id}/trace")
def get_process_trace(application_id: str):
    path = _trace_path(application_id)
    if not os.path.exists(path):
        return {"trace": []}
    with open(path, "r") as f:
        return {"trace": json.load(f)}

@app.post("/score/{application_id}")
def score_application_endpoint(application_id: str):
    """Score an application using ML model with SHAP explanations."""
    db = SessionLocal()
    try:
        from backend.fastapi_app.ml.scoring_service import score_application

        # Get application data
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}

        # Check if validation exists
        validation_path = os.path.join(APPS_DIR, application_id, "validation.json")
        if os.path.exists(validation_path):
            with open(validation_path, "r") as f:
                validation_data = json.load(f)
        else:
            validation_data = {}

        # Prepare data for ML scoring
        application_data = {
            'app_row': {
                'declared_monthly_income': app_row.declared_monthly_income,
                'household_size': app_row.household_size
            },
            'validation': validation_data
        }

        # Get ML scoring results
        ml_result = score_application(application_data)

        # Update application status in database
        decision = ml_result['decision']
        confidence = ml_result['confidence']

        # Create reason from top factors
        top_factors = ml_result.get('top_factors', [])
        factor_descriptions = []
        for factor in top_factors[:2]:
            factor_descriptions.append(
                f"{factor['feature']}={factor['value']:.1f} ({'positive' if factor['impact'] > 0 else 'negative'} impact)"
            )

        reason = f"ML Decision: {decision} (confidence: {confidence:.2f}). Key factors: {', '.join(factor_descriptions)}"

        app_row.decision = decision
        app_row.decision_reason = reason
        app_row.status = "ML_SCORED"
        db.add(app_row); db.commit()

        # Save ML results to file
        dest_dir = os.path.join(APPS_DIR, application_id)
        os.makedirs(dest_dir, exist_ok=True)
        ml_score_path = os.path.join(dest_dir, "ml_score.json")
        with open(ml_score_path, "w") as f:
            json.dump(ml_result, f, indent=2)

        # Audit log
        _audit(db, application_id, "ml_score", {"decision": decision, "confidence": confidence})

        return {"ok": True, "ml_result": ml_result, "application_id": application_id}

    except Exception as e:
        return {"ok": False, "error": str(e), "application_id": application_id}
    finally:
        db.close()

@app.get("/score/{application_id}")
def get_ml_score(application_id: str):
    """Get saved ML scoring results for an application."""
    score_path = os.path.join(APPS_DIR, application_id, "ml_score.json")
    if not os.path.exists(score_path):
        return {"error": "ml_score_not_found", "application_id": application_id}

    with open(score_path, "r") as f:
        ml_result = json.load(f)

    return {"ok": True, "ml_result": ml_result, "application_id": application_id}

@app.post("/graph/analyze/{application_id}")
def analyze_graph(application_id: str):
    """Analyze application entities and populate graph database."""
    db = SessionLocal()
    try:
        from backend.fastapi_app.graph.entity_extraction import get_entity_extractor
        from backend.fastapi_app.graph.entity_normalization import get_entity_normalizer
        from backend.fastapi_app.graph.neo4j_service import get_neo4j_service, ensure_neo4j_connection

        # Get application data
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}

        docs = db.query(Document).filter(Document.application_id == application_id).all()

        # Load existing extraction data if available
        ocr_data = {}
        parsed_data = {}
        dest_dir = os.path.join(APPS_DIR, application_id)

        ocr_path = os.path.join(dest_dir, "ocr.json")
        if os.path.exists(ocr_path):
            with open(ocr_path, "r") as f:
                ocr_data = json.load(f)

        parsed_path = os.path.join(dest_dir, "parsed.json")
        if os.path.exists(parsed_path):
            with open(parsed_path, "r") as f:
                parsed_data = json.load(f)

        # Prepare application data for entity extraction
        application_data = {
            'app_row': {
                'name': app_row.name,
                'emirates_id': app_row.emirates_id,
                'declared_monthly_income': app_row.declared_monthly_income,
                'household_size': app_row.household_size,
                'channel': app_row.channel,
                'submitted_at': app_row.submitted_at.isoformat() if app_row.submitted_at else None
            },
            'doc_paths': [d.path for d in docs if os.path.exists(d.path)],
            'ocr': ocr_data,
            'parsed': parsed_data
        }

        # Extract entities
        extractor = get_entity_extractor()
        entities = extractor.extract_all_entities(application_data)

        # Normalize and deduplicate entities
        normalizer = get_entity_normalizer()
        deduplicated_entities = normalizer.deduplicate_entities(entities)

        # Store in MongoDB for analytics and fast retrieval
        mongo_service = get_mongo_service()
        mongo_service.store_application_data(application_id, application_data)

        # Populate graph
        graph_results = {'entities_extracted': len(entities), 'entities_deduplicated': len(deduplicated_entities)}

        try:
            if ensure_neo4j_connection():
                neo4j_service = get_neo4j_service()
                graph_results = neo4j_service.populate_graph_from_entities(
                    application_id, deduplicated_entities, application_data
                )
            else:
                graph_results['warning'] = 'Neo4j not available - entities extracted but not stored in graph'
        except Exception as graph_error:
            graph_results['warning'] = f'Graph population failed: {str(graph_error)}'

        # Save results
        os.makedirs(dest_dir, exist_ok=True)
        analysis_path = os.path.join(dest_dir, "graph_analysis.json")
        with open(analysis_path, "w") as f:
            analysis_data = {
                'entities': [
                    {
                        'type': e.entity_type,
                        'value': e.value,
                        'normalized': e.normalized,
                        'confidence': e.confidence,
                        'source': e.source
                    } for e in deduplicated_entities
                ],
                'graph_results': graph_results
            }
            json.dump(analysis_data, f, indent=2)

        # Audit log
        _audit(db, application_id, "graph_analysis", graph_results)

        return {
            "ok": True,
            "application_id": application_id,
            "entities_extracted": len(entities),
            "entities_deduplicated": len(deduplicated_entities),
            "graph_results": graph_results
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "application_id": application_id}
    finally:
        db.close()

@app.get("/graph/analysis/{application_id}")
def get_graph_analysis(application_id: str):
    """Get saved graph analysis results for an application."""
    analysis_path = os.path.join(APPS_DIR, application_id, "graph_analysis.json")
    if not os.path.exists(analysis_path):
        return {"error": "graph_analysis_not_found", "application_id": application_id}

    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)

    return {"ok": True, "analysis": analysis_data, "application_id": application_id}

@app.post("/duplicates/detect/{application_id}")
def detect_duplicates(application_id: str):
    """Detect duplicate applications for a given application."""
    db = SessionLocal()
    try:
        from backend.fastapi_app.graph.duplicate_detection import get_duplicate_detector

        # Get application data
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}

        # Get historical applications for comparison
        detector = get_duplicate_detector()
        historical_apps = db.query(Application).filter(
            Application.id != application_id,
            Application.submitted_at >= app_row.submitted_at - dt.timedelta(days=detector.duplicate_time_window)
        ).all()

        historical_data = [
            {
                'application_id': h.id,
                'submitted_at': h.submitted_at.isoformat() if h.submitted_at else None,
                'name': h.name,
                'emirates_id': h.emirates_id
            } for h in historical_apps
        ]

        # Find duplicates
        duplicates = detector.find_duplicate_applications(application_id, historical_data)
        statistics = detector.get_duplicate_statistics(application_id, duplicates)

        # Format results
        duplicate_results = {
            'duplicates_found': len(duplicates),
            'duplicates': [
                {
                    'duplicate_application_id': d.duplicate_application_id,
                    'match_type': d.match_type,
                    'similarity_score': d.similarity_score,
                    'confidence': d.confidence,
                    'risk_level': d.risk_level,
                    'evidence': d.evidence
                } for d in duplicates
            ],
            'statistics': statistics
        }

        # Audit log
        _audit(db, application_id, "duplicate_detection", duplicate_results)

        return {"ok": True, "results": duplicate_results, "application_id": application_id}

    except Exception as e:
        return {"ok": False, "error": str(e), "application_id": application_id}
    finally:
        db.close()

@app.post("/conflicts/detect/{application_id}")
def detect_conflicts(application_id: str):
    """Detect data conflicts within an application."""
    db = SessionLocal()
    try:
        from backend.fastapi_app.graph.duplicate_detection import get_duplicate_detector

        # Get application data
        app_row = db.query(Application).filter(Application.id == application_id).first()
        if not app_row:
            return {"error": "not_found", "application_id": application_id}

        # Load existing validation data
        validation_data = {}
        validation_path = os.path.join(APPS_DIR, application_id, "validation.json")
        if os.path.exists(validation_path):
            with open(validation_path, "r") as f:
                validation_data = json.load(f)

        # Prepare application data
        application_data = {
            'app_row': {
                'declared_monthly_income': app_row.declared_monthly_income,
                'household_size': app_row.household_size
            },
            'validation': validation_data,
            'application_id': application_id
        }

        # Detect conflicts
        detector = get_duplicate_detector()
        conflicts = detector.detect_data_conflicts(application_id, application_data)

        # Format results
        conflict_results = {
            'conflicts_found': len(conflicts),
            'conflicts': [
                {
                    'type': c.conflict_type,
                    'severity': c.severity,
                    'description': c.description,
                    'evidence': c.evidence,
                    'suggestions': c.suggestions
                } for c in conflicts
            ]
        }

        # Audit log
        _audit(db, application_id, "conflict_detection", conflict_results)

        return {"ok": True, "results": conflict_results, "application_id": application_id}

    except Exception as e:
        return {"ok": False, "error": str(e), "application_id": application_id}
    finally:
        db.close()

@app.get("/duplicates/{application_id}")
def get_duplicates(application_id: str):
    """Get saved duplicate detection results for an application."""
    analysis_path = os.path.join(APPS_DIR, application_id, "graph_analysis.json")
    if not os.path.exists(analysis_path):
        return {"error": "analysis_not_found", "application_id": application_id}

    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)

    duplicate_data = analysis_data.get('duplicate_detection', {})
    if not duplicate_data:
        return {"error": "duplicates_not_found", "application_id": application_id}

    return {"ok": True, "results": duplicate_data, "application_id": application_id}

@app.get("/conflicts/{application_id}")
def get_conflicts(application_id: str):
    """Get saved conflict detection results for an application."""
    analysis_path = os.path.join(APPS_DIR, application_id, "graph_analysis.json")
    if not os.path.exists(analysis_path):
        return {"error": "analysis_not_found", "application_id": application_id}

    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)

    conflict_data = analysis_data.get('conflict_detection', {})
    if not conflict_data:
        return {"error": "conflicts_not_found", "application_id": application_id}

    return {"ok": True, "results": conflict_data, "application_id": application_id}

# Phase 6: Chat and RAG endpoints
@app.post("/chat/session")
def create_chat_session():
    """Create a new chat session."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()
        session_id = chat_service.create_session()
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat/{session_id}/message")
def send_chat_message(session_id: str, message: dict):
    """Send a message in a chat session."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()

        content = message.get("content", "")
        message_type = message.get("type", "text")

        result = chat_service.process_user_message(session_id, content, message_type)
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat/{session_id}/upload")
def upload_chat_document(session_id: str, file: UploadFile = File(...)):
    """Upload a document during chat conversation."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()

        # Save uploaded file
        session = chat_service.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}

        # Create temporary directory for this session if needed
        session_dir = os.path.join(APPS_DIR, f"chat_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Save file
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Process upload
        result = chat_service.upload_document(session_id, file_path, file.filename)
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/chat/{session_id}/messages")
def get_chat_messages(session_id: str):
    """Get messages for a chat session."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()
        messages = chat_service.get_messages(session_id)
        return {"ok": True, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/chat/{session_id}/status")
def get_chat_status(session_id: str):
    """Get chat session status and progress."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()

        session = chat_service.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}

        return {
            "ok": True,
            "session_id": session_id,
            "state": session.state.value,
            "data_collected": chat_service._get_progress_summary(session),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat/{session_id}/process")
def process_chat_application(session_id: str):
    """Process the application created through chat."""
    try:
        from backend.fastapi_app.chat.chat_service import get_chat_service
        chat_service = get_chat_service()
        result = chat_service.process_application_with_orchestrator(session_id)
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/rag/initialize")
def initialize_knowledge_base():
    """Initialize the RAG knowledge base."""
    try:
        from backend.fastapi_app.rag.rag_service import initialize_knowledge_base
        success = initialize_knowledge_base()
        if success:
            return {"ok": True, "message": "Knowledge base initialized successfully"}
        else:
            return {"ok": False, "error": "Failed to initialize knowledge base"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/rag/search")
def search_knowledge_base(query: str, limit: int = 5):
    """Search the knowledge base."""
    try:
        from backend.fastapi_app.rag.rag_service import get_rag_service
        rag_service = get_rag_service()
        results = rag_service.search_knowledge(query, limit)
        return {"ok": True, "results": [{"content": r.content, "score": r.score, "category": r.category} for r in results]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/rag/stats")
def get_knowledge_base_stats():
    """Get knowledge base statistics."""
    try:
        from backend.fastapi_app.rag.rag_service import get_rag_service
        rag_service = get_rag_service()
        stats = rag_service.get_knowledge_stats()
        return {"ok": True, "stats": stats}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ReAct Framework Endpoints
@app.get("/react/config")
def get_react_config():
    """Get current ReAct configuration."""
    try:
        from backend.fastapi_app.reasoning.react_config import get_config_dict
        return {"ok": True, "config": get_config_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/react/config")
def update_react_config(config_updates: dict):
    """Update ReAct configuration."""
    try:
        from backend.fastapi_app.reasoning.react_config import update_react_config
        updated_config = update_react_config(**config_updates)
        return {"ok": True, "config": {
            "enabled": updated_config.enabled,
            "max_iterations": updated_config.max_iterations,
            "timeout_seconds": updated_config.timeout_seconds,
            "use_for_complex_cases_only": updated_config.use_for_complex_cases_only,
            "debug_mode": updated_config.debug_mode,
            "save_traces": updated_config.save_traces
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/react/tools")
def get_react_tools():
    """Get available ReAct tools."""
    try:
        from backend.fastapi_app.reasoning.tools.tool_initializer import get_tool_summary
        summary = get_tool_summary()
        return {"ok": True, "tools": summary}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/react/trace/{application_id}")
def get_react_trace(application_id: str):
    """Get ReAct reasoning trace for an application."""
    try:
        from backend.fastapi_app.reasoning import get_react_summary
        trace = get_react_summary(application_id)
        if trace:
            return {"ok": True, "trace": trace}
        else:
            return {"ok": False, "error": "No ReAct trace found for this application"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
