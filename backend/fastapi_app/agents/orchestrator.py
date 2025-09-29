from __future__ import annotations

import os
import json
import sqlite3
import datetime as dt
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
# If you need async: from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.fastapi_app.main import (
    SessionLocal,
    Application,
    Document,
    APPS_DIR,
    DATA_DIR,
)
from backend.fastapi_app.services.ocr_service import run_ocr
from backend.fastapi_app.services.parse_service import parse_document
from backend.fastapi_app.services.validation_service import validate_application
# If you want to read thread/run IDs inside nodes later:
# from langchain_core.runnables.config import RunnableConfig


class OrchestratorState(TypedDict, total=False):
    app_id: str
    status: str
    notes: List[str]
    ocr: Dict[str, Any]
    parsed: Dict[str, Any]
    extraction_summary: Dict[str, Any]
    validation: Dict[str, Any]
    graph_analysis: Dict[str, Any]
    eligibility: Dict[str, Any]


def _trace_path(app_id: str) -> str:
    dest_dir = os.path.join(APPS_DIR, app_id)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, "orchestrator_trace.json")


def append_trace(app_id: str, event: Dict[str, Any]) -> None:
    path = _trace_path(app_id)
    now = dt.datetime.utcnow().isoformat()
    event = {"ts": now, **event}
    arr = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                arr = json.load(f)
        except Exception:
            arr = []
    arr.append(event)
    with open(path, "w") as f:
        json.dump(arr, f, indent=2)


def load_app(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        app_row = db.query(Application).filter(Application.id == app_id).first()
        if not app_row:
            raise ValueError(f"application not_found: {app_id}")
        docs = db.query(Document).filter(Document.application_id == app_id).all()
        state["status"] = "LOADED"
        state.setdefault("notes", []).append(f"Loaded app with {len(docs)} docs.")
        append_trace(app_id, {"step": "load_app", "status": "ok", "docs": len(docs)})
        return state
    except Exception as e:
        append_trace(app_id, {"step": "load_app", "status": "error", "error": str(e)})
        raise
    finally:
        db.close()


def extract_node(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.application_id == app_id).all()
        dest_dir = os.path.join(APPS_DIR, app_id)
        ocr_dir = os.path.join(dest_dir, "ocr")
        os.makedirs(ocr_dir, exist_ok=True)

        ocr_reports: Dict[str, Any] = {}
        parse_reports: Dict[str, Any] = {}

        for d in docs:
            ocr_reports[d.filename] = run_ocr(d.path, ocr_dir)
            parse_reports[d.filename] = parse_document(d.path)

        with open(os.path.join(dest_dir, "ocr.json"), "w") as f:
            json.dump(ocr_reports, f, indent=2)
        with open(os.path.join(dest_dir, "parsed.json"), "w") as f:
            json.dump(parse_reports, f, indent=2)

        total_inflow = sum(
            (r.get("total_inflow", 0.0) for r in parse_reports.values() if isinstance(r, dict))
        )
        total_outflow = sum(
            (r.get("total_outflow", 0.0) for r in parse_reports.values() if isinstance(r, dict))
        )
        summary = {
            "application_id": app_id,
            "total_inflow": float(total_inflow),
            "total_outflow": float(total_outflow),
            "documents_processed": len(docs),
            "notes": "OCR/parsing best-effort; missing system binaries reported as warnings.",
        }
        with open(os.path.join(dest_dir, "extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        state["ocr"] = ocr_reports
        state["parsed"] = parse_reports
        state["extraction_summary"] = summary
        state["status"] = "EXTRACTED"
        append_trace(app_id, {"step": "extract", "status": "ok", "summary": summary})
        return state
    except Exception as e:
        append_trace(app_id, {"step": "extract", "status": "error", "error": str(e)})
        raise
    finally:
        db.close()


def validate_node(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        app_row = db.query(Application).filter(Application.id == app_id).first()
        docs = db.query(Document).filter(Document.application_id == app_id).all()
        doc_paths = [d.path for d in docs if os.path.exists(d.path)]

        result = validate_application(
            app_row={
                "declared_monthly_income": app_row.declared_monthly_income,
                "household_size": app_row.household_size,
            },
            doc_paths=doc_paths,
        )
        with open(os.path.join(APPS_DIR, app_id, "validation.json"), "w") as f:
            json.dump(result, f, indent=2)

        app_row.status = "VALIDATED"
        db.add(app_row)
        db.commit()

        state["validation"] = result
        state["status"] = "VALIDATED"
        append_trace(app_id, {"step": "validate", "status": "ok", "validation": result})
        return state
    except Exception as e:
        append_trace(app_id, {"step": "validate", "status": "error", "error": str(e)})
        raise
    finally:
        db.close()


def eligibility_node(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        from backend.fastapi_app.ml.scoring_service import score_application

        app_row = db.query(Application).filter(Application.id == app_id).first()

        application_data = {
            "app_row": {
                "declared_monthly_income": app_row.declared_monthly_income,
                "household_size": app_row.household_size,
            },
            "validation": state.get("validation", {}) or {},
        }

        ml_result = score_application(application_data)

        decision = ml_result["decision"]
        confidence = ml_result["confidence"]

        top_factors = ml_result.get("top_factors", []) or []
        factor_descriptions = []
        for factor in top_factors[:2]:  # only top-2 for the reason string
            val = factor.get("value")
            val_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
            factor_descriptions.append(
                f"{factor.get('feature')}={val_str} "
                f"({'positive' if (factor.get('impact', 0) > 0) else 'negative'} impact)"
            )

        reason = (
            f"ML Decision: {decision} (confidence: {confidence:.2f}). "
            f"Key factors: {', '.join(factor_descriptions)}"
        )

        # Persist to DB
        app_row.decision = decision
        app_row.decision_reason = reason
        app_row.status = "DECIDED"
        db.add(app_row)
        db.commit()

        # Save full ML output for audit
        dest_dir = os.path.join(APPS_DIR, app_id)
        with open(os.path.join(dest_dir, "ml_score.json"), "w") as f:
            json.dump(ml_result, f, indent=2)

        state["eligibility"] = {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "ml_results": ml_result,
        }
        state["status"] = "DECIDED"
        append_trace(app_id, {"step": "ml_eligibility", "status": "ok", "decision": state["eligibility"]})
        return state
    except Exception as e:
        append_trace(app_id, {"step": "ml_eligibility", "status": "error", "error": str(e)})
        raise
    finally:
        db.close()


def graph_analysis_node(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        from backend.fastapi_app.graph.entity_extraction import get_entity_extractor
        from backend.fastapi_app.graph.entity_normalization import get_entity_normalizer
        from backend.fastapi_app.graph.neo4j_service import get_neo4j_service, ensure_neo4j_connection
        from backend.fastapi_app.graph.duplicate_detection import get_duplicate_detector

        app_row = db.query(Application).filter(Application.id == app_id).first()
        docs = db.query(Document).filter(Document.application_id == app_id).all()

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
            'ocr': state.get('ocr', {}),
            'parsed': state.get('parsed', {}),
            'validation': state.get('validation', {})
        }

        # Extract entities
        extractor = get_entity_extractor()
        entities = extractor.extract_all_entities(application_data)

        # Normalize and deduplicate entities
        normalizer = get_entity_normalizer()
        deduplicated_entities = normalizer.deduplicate_entities(entities)

        # Populate graph (try to connect to Neo4j, but don't fail if unavailable)
        graph_results = {'entities_extracted': len(entities), 'entities_deduplicated': len(deduplicated_entities)}

        try:
            if ensure_neo4j_connection():
                neo4j_service = get_neo4j_service()
                graph_results = neo4j_service.populate_graph_from_entities(
                    app_id, deduplicated_entities, application_data
                )
            else:
                graph_results['warning'] = 'Neo4j not available - entities extracted but not stored in graph'
        except Exception as graph_error:
            graph_results['warning'] = f'Graph population failed: {str(graph_error)}'

        # Duplicate detection and conflict analysis
        duplicate_results = {}
        conflict_results = {}

        try:
            detector = get_duplicate_detector()

            # Detect data conflicts within this application
            conflicts = detector.detect_data_conflicts(app_id, application_data)
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

            # Get historical applications for duplicate detection
            historical_apps = db.query(Application).filter(
                Application.id != app_id,
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

            # Find duplicate applications
            duplicates = detector.find_duplicate_applications(app_id, historical_data)
            duplicate_stats = detector.get_duplicate_statistics(app_id, duplicates)

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
                'statistics': duplicate_stats
            }

        except Exception as detection_error:
            duplicate_results['error'] = f'Duplicate detection failed: {str(detection_error)}'
            conflict_results['error'] = f'Conflict detection failed: {str(detection_error)}'

        # Save comprehensive graph analysis results
        dest_dir = os.path.join(APPS_DIR, app_id)
        with open(os.path.join(dest_dir, "graph_analysis.json"), "w") as f:
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
                'graph_results': graph_results,
                'duplicate_detection': duplicate_results,
                'conflict_detection': conflict_results
            }
            json.dump(analysis_data, f, indent=2)

        state["graph_analysis"] = {
            'entities_count': len(deduplicated_entities),
            'graph_results': graph_results,
            'duplicate_detection': duplicate_results,
            'conflict_detection': conflict_results
        }
        state["status"] = "GRAPH_ANALYZED"

        # Enhanced trace logging
        trace_data = {
            "entities": len(deduplicated_entities),
            "graph": graph_results,
            "duplicates": duplicate_results.get('duplicates_found', 0),
            "conflicts": conflict_results.get('conflicts_found', 0)
        }
        append_trace(app_id, {"step": "graph_analysis", "status": "ok", "results": trace_data})
        return state

    except Exception as e:
        append_trace(app_id, {"step": "graph_analysis", "status": "error", "error": str(e)})
        # Don't fail the entire pipeline if graph analysis fails
        state["graph_analysis"] = {'error': str(e)}
        state["status"] = "GRAPH_ANALYSIS_FAILED"
        return state
    finally:
        db.close()

def finalize_node(state: OrchestratorState) -> OrchestratorState:
    app_id = state["app_id"]
    state["status"] = "DONE"
    append_trace(app_id, {"step": "finalize", "status": "ok"})
    return state


def _checkpointer_path() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "orchestrator.db")


def _make_checkpointer() -> SqliteSaver:
    """
    Create a *real* SqliteSaver (not a context manager proxy) by
    passing a sqlite3 connection we own to SqliteSaver(...).
    """
    path = _checkpointer_path()
    # For FastAPI multi-threading, allow cross-thread use of this connection.
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("load_app", load_app)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("graph_analysis", graph_analysis_node)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("load_app")
    graph.add_edge("load_app", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "graph_analysis")
    graph.add_edge("graph_analysis", "eligibility")
    graph.add_edge("eligibility", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = _make_checkpointer()  # <-- fixed (no context-manager proxy)
    app = graph.compile(checkpointer=checkpointer)
    return app


GRAPH_APP = build_graph()


def run_orchestration(app_id: str) -> Dict[str, Any]:
    initial: OrchestratorState = {"app_id": app_id, "status": "NEW", "notes": []}
    # Thread persistence per LangGraph docs
    config = {"configurable": {"thread_id": f"proc-{app_id}"}}
    final_state = GRAPH_APP.invoke(initial, config)

    path = _trace_path(app_id)
    trace = []
    if os.path.exists(path):
        with open(path, "r") as f:
            trace = json.load(f)

    return {"state": final_state, "trace": trace}


__all__ = ["run_orchestration", "_trace_path", "append_trace"]
