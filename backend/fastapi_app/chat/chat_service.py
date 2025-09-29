"""
Chat service for Phase 6 conversational application processing.
Handles conversation flow, document processing, and decision making with RAG integration.
"""

import os
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from ..rag.rag_service import get_rag_service


class ConversationState(Enum):
    """States in the conversation flow."""
    GREETING = "greeting"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_EMIRATES_ID = "collecting_emirates_id"
    COLLECTING_INCOME = "collecting_income"
    COLLECTING_HOUSEHOLD_SIZE = "collecting_household_size"
    COLLECTING_DOCUMENTS = "collecting_documents"
    PROCESSING_APPLICATION = "processing_application"
    SHOWING_RESULTS = "showing_results"
    COMPLETED = "completed"


@dataclass
class ConversationData:
    """Data collected during conversation."""
    name: Optional[str] = None
    emirates_id: Optional[str] = None
    declared_monthly_income: Optional[float] = None
    household_size: Optional[int] = None
    channel: str = "Chat"
    documents: List[str] = None
    application_id: Optional[str] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = []


@dataclass
class ChatMessage:
    """Represents a chat message."""
    id: str
    sender: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_type: str = "text"  # 'text', 'file', 'system'
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatSession:
    """Represents a chat session."""
    session_id: str
    state: ConversationState
    data: ConversationData
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class ChatService:
    """Service for managing conversational application processing."""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.sessions: Dict[str, ChatSession] = {}

    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = ChatSession(
            session_id=session_id,
            state=ConversationState.GREETING,
            data=ConversationData(),
            messages=[],
            created_at=now,
            updated_at=now
        )

        self.sessions[session_id] = session

        # Send greeting message
        greeting_msg = self._generate_greeting_message()
        self._add_message(session_id, "assistant", greeting_msg, "text")

        return session_id

    def process_user_message(self, session_id: str, message: str, message_type: str = "text") -> Dict[str, Any]:
        """Process a user message and return the response."""
        if session_id not in self.sessions:
            return {"error": "Session not found", "session_id": session_id}

        session = self.sessions[session_id]

        # Add user message
        self._add_message(session_id, "user", message, message_type)

        # Process based on current state
        response = self._process_state(session, message, message_type)

        # Add assistant response to message history
        self._add_message(session_id, "assistant", response, "text")

        # Update session timestamp
        session.updated_at = datetime.utcnow()

        return {
            "session_id": session_id,
            "response": response,
            "state": session.state.value,
            "data_collected": self._get_progress_summary(session)
        }

    def upload_document(self, session_id: str, file_path: str, filename: str) -> Dict[str, Any]:
        """Handle document upload during conversation."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]
        session.data.documents.append(file_path)

        # Process document immediately
        processing_result = self._process_document_upload(session, file_path, filename)

        # Add system message about document processing
        self._add_message(session_id, "assistant", processing_result["message"], "system")

        return processing_result

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return self.sessions.get(session_id)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        messages = []
        for msg in session.messages:
            msg_dict = asdict(msg)
            # Convert datetime to ISO string for JSON serialization
            if isinstance(msg_dict['timestamp'], datetime):
                msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
            messages.append(msg_dict)
        return messages

    def _generate_greeting_message(self) -> str:
        """Generate greeting message with RAG context."""
        # Get RAG context for greeting (not used in basic greeting but available for enhancement)
        try:
            self.rag_service.get_contextual_response(
                "greeting social support application process"
            )
        except Exception:
            pass  # Fallback to basic greeting if RAG fails

        return (
            "Hello! 👋 I'm your AI assistant for UAE Social Support applications.\n\n"
            "I'll help you through the entire application process step by step. "
            "I can process your documents in real-time and provide instant feedback.\n\n"
            "To get started, could you please tell me your full name as it appears on your Emirates ID?"
        )

    def _process_state(self, session: ChatSession, message: str, message_type: str) -> str:
        """Process message based on current conversation state."""
        if session.state == ConversationState.GREETING:
            return self._handle_name_collection(session, message)

        elif session.state == ConversationState.COLLECTING_NAME:
            return self._handle_name_collection(session, message)

        elif session.state == ConversationState.COLLECTING_EMIRATES_ID:
            return self._handle_emirates_id_collection(session, message)

        elif session.state == ConversationState.COLLECTING_INCOME:
            return self._handle_income_collection(session, message)

        elif session.state == ConversationState.COLLECTING_HOUSEHOLD_SIZE:
            return self._handle_household_size_collection(session, message)

        elif session.state == ConversationState.COLLECTING_DOCUMENTS:
            return self._handle_document_collection(session, message)

        elif session.state == ConversationState.PROCESSING_APPLICATION:
            return "Please wait while I process your application..."

        else:
            return "I'm not sure how to help with that. Could you please clarify?"

    def _handle_name_collection(self, session: ChatSession, message: str) -> str:
        """Handle name collection."""
        # Simple name validation
        if len(message.strip()) < 2:
            return "Please provide your full name as it appears on your Emirates ID."

        session.data.name = message.strip()
        session.state = ConversationState.COLLECTING_EMIRATES_ID

        return (
            f"Thank you, {session.data.name}! ✓\n\n"
            "Now I need your Emirates ID number. Please enter it in the format: "
            "784-XXXX-XXXXXXX-X (15 digits total)."
        )

    def _handle_emirates_id_collection(self, session: ChatSession, message: str) -> str:
        """Handle Emirates ID collection with validation."""
        emirates_id = message.strip().replace(" ", "").replace("-", "")

        # Basic Emirates ID validation
        if len(emirates_id) != 15 or not emirates_id.isdigit():
            return (
                "Please provide a valid Emirates ID number (15 digits). "
                "Format: 784-XXXX-XXXXXXX-X"
            )

        # Format the Emirates ID
        formatted_id = f"{emirates_id[:3]}-{emirates_id[3:7]}-{emirates_id[7:14]}-{emirates_id[14]}"
        session.data.emirates_id = formatted_id
        session.state = ConversationState.COLLECTING_INCOME

        return (
            f"Emirates ID received: {formatted_id} ✓\n\n"
            "What is your total monthly household income in AED? "
            "Please include all sources of income (salary, allowances, etc.)."
        )

    def _handle_income_collection(self, session: ChatSession, message: str) -> str:
        """Handle income collection with validation."""
        try:
            # Extract numeric value
            income_str = message.strip().replace("AED", "").replace(",", "").replace(" ", "")
            income = float(income_str)

            if income < 0:
                return "Income cannot be negative. Please provide your monthly household income in AED."

            if income > 100000:
                return "That seems quite high. Are you sure that's your monthly income in AED?"

            session.data.declared_monthly_income = income
            session.state = ConversationState.COLLECTING_HOUSEHOLD_SIZE

            # Get RAG context about income thresholds
            rag_result = self.rag_service.get_contextual_response(
                f"income threshold household income {income} AED eligibility"
            )

            response = f"Monthly income: {income:,.0f} AED ✓\n\n"

            if rag_result["confidence"] > 0.7:
                response += f"📋 {rag_result['response']}\n\n"

            response += "How many people are in your household (including yourself)?"

            return response

        except ValueError:
            return "Please provide a valid income amount in AED (numbers only)."

    def _handle_household_size_collection(self, session: ChatSession, message: str) -> str:
        """Handle household size collection."""
        try:
            household_size = int(message.strip())

            if household_size < 1:
                return "Household size must be at least 1 person."

            if household_size > 20:
                return "That seems like a very large household. Please confirm the number of people."

            session.data.household_size = household_size
            session.state = ConversationState.COLLECTING_DOCUMENTS

            # Get RAG context about required documents
            rag_result = self.rag_service.get_contextual_response(
                "required documents social support application UAE"
            )

            response = f"Household size: {household_size} people ✓\n\n"
            response += "📄 **Required Documents:**\n"

            if rag_result["confidence"] > 0.7:
                response += f"{rag_result['response']}\n\n"
            else:
                response += (
                    "• Emirates ID copy\n"
                    "• Salary certificate or employment letter\n"
                    "• Bank statements (last 3 months)\n"
                    "• Utility bills\n"
                    "• Tenancy contract\n\n"
                )

            response += (
                "Please upload your documents one by one. I'll process each document "
                "immediately and let you know if everything looks good.\n\n"
                "You can type 'done' when you've uploaded all required documents."
            )

            return response

        except ValueError:
            return "Please provide a valid number for household size."

    def _handle_document_collection(self, session: ChatSession, message: str) -> str:
        """Handle document collection phase."""
        if message.lower().strip() == "done":
            return self._initiate_processing(session)

        return (
            "Please upload your documents using the file upload feature, "
            "or type 'done' when you've finished uploading all required documents."
        )

    def _process_document_upload(self, session: ChatSession, file_path: str, filename: str) -> Dict[str, Any]:
        """Process uploaded document with real-time feedback."""
        try:
            # Import here to avoid circular imports
            from ..services.ocr_service import run_ocr
            from ..services.parse_service import parse_document

            # Determine document type and provide specific feedback
            doc_type = self._classify_document(filename)

            # Run OCR/parsing
            ocr_result = run_ocr(file_path, os.path.dirname(file_path))
            parse_result = parse_document(file_path)

            # Analyze results and provide feedback
            feedback = self._analyze_document_results(doc_type, ocr_result, parse_result, session.data)

            return {
                "success": True,
                "document_type": doc_type,
                "message": feedback,
                "filename": filename
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Failed to process {filename}: {str(e)}",
                "filename": filename
            }

    def _classify_document(self, filename: str) -> str:
        """Classify document type based on filename."""
        filename_lower = filename.lower()

        if "emirates" in filename_lower or "id" in filename_lower:
            return "emirates_id"
        elif "salary" in filename_lower or "employment" in filename_lower:
            return "employment"
        elif "bank" in filename_lower or "statement" in filename_lower:
            return "bank_statement"
        elif "utility" in filename_lower or "bill" in filename_lower:
            return "utility_bill"
        elif "tenancy" in filename_lower or "contract" in filename_lower:
            return "tenancy_contract"
        else:
            return "unknown"

    def _analyze_document_results(self, doc_type: str, ocr_result: Dict, parse_result: Dict,
                                 conversation_data: ConversationData) -> str:
        """Analyze document processing results and provide feedback."""
        feedback = f"✅ **{doc_type.replace('_', ' ').title()} processed successfully!**\n\n"

        # Check for data consistency
        if doc_type == "emirates_id" and ocr_result.get("ok"):
            # Check if extracted name matches conversation data
            feedback += "• Emirates ID verified\n"
            feedback += "• Identity information extracted\n"

        elif doc_type == "bank_statement" and parse_result.get("ok"):
            # Check income consistency
            extracted_income = parse_result.get("total_inflow", 0)
            if extracted_income > 0 and conversation_data.declared_monthly_income:
                variance = abs(extracted_income - conversation_data.declared_monthly_income) / conversation_data.declared_monthly_income
                if variance > 0.25:
                    feedback += "⚠️ **Income discrepancy detected:**\n"
                    feedback += f"• Declared: {conversation_data.declared_monthly_income:,.0f} AED\n"
                    feedback += f"• Bank statement: {extracted_income:,.0f} AED\n"
                    feedback += "Please be ready to explain this difference.\n\n"
                else:
                    feedback += f"• Income verified: {extracted_income:,.0f} AED ✓\n"

        else:
            if not ocr_result.get("ok") and not parse_result.get("ok"):
                feedback = f"⚠️ **{doc_type.replace('_', ' ').title()} uploaded but needs review:**\n"
                feedback += "• Document format may need manual verification\n"
                feedback += "• Please ensure document is clear and readable\n\n"

        return feedback

    def _initiate_processing(self, session: ChatSession) -> str:
        """Initiate full application processing."""
        if not session.data.documents:
            return (
                "⚠️ You haven't uploaded any documents yet. "
                "Please upload at least your Emirates ID and one supporting document before proceeding."
            )

        session.state = ConversationState.PROCESSING_APPLICATION

        # Create application in the system
        try:
            application_id = self._create_application(session.data)
            session.data.application_id = application_id
            session.state = ConversationState.SHOWING_RESULTS

            return (
                "🎉 **Application Created Successfully!**\n\n"
                f"**Application ID:** {application_id}\n\n"
                "I'm now processing your application through our automated system. "
                "This includes:\n"
                "• Document verification\n"
                "• Income assessment\n"
                "• Eligibility evaluation\n"
                "• ML-powered decision making\n\n"
                "⏳ Processing will take a few moments..."
            )

        except Exception as e:
            return f"❌ Failed to create application: {str(e)}"

    def _create_application(self, data: ConversationData) -> str:
        """Create application using existing system."""
        # Import here to avoid circular imports
        from ..database import DatabaseSession, Application, Document
        import uuid

        try:
            with DatabaseSession() as db:
                app_id = str(uuid.uuid4())
                app_row = Application(
                    id=app_id,
                    name=data.name,
                    emirates_id=data.emirates_id,
                    declared_monthly_income=data.declared_monthly_income,
                    household_size=data.household_size,
                    channel=data.channel,
                    status="RECEIVED",
                )
                db.add(app_row)

                # Add documents
                for doc_path in data.documents:
                    doc = Document(
                        id=str(uuid.uuid4()),
                        application_id=app_id,
                        path=doc_path,
                        filename=os.path.basename(doc_path)
                    )
                    db.add(doc)

                # Session automatically commits/rollbacks via context manager
                return app_id

    def _get_progress_summary(self, session: ChatSession) -> Dict[str, Any]:
        """Get summary of data collection progress."""
        data = session.data
        return {
            "name": data.name is not None,
            "emirates_id": data.emirates_id is not None,
            "income": data.declared_monthly_income is not None,
            "household_size": data.household_size is not None,
            "documents": len(data.documents),
            "application_id": data.application_id
        }

    def _add_message(self, session_id: str, sender: str, content: str, message_type: str):
        """Add message to session."""
        if session_id not in self.sessions:
            return

        message = ChatMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            content=content,
            timestamp=datetime.utcnow(),
            message_type=message_type
        )

        self.sessions[session_id].messages.append(message)

    def process_application_with_orchestrator(self, session_id: str) -> Dict[str, Any]:
        """Process application using existing orchestrator."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]
        if not session.data.application_id:
            return {"error": "No application ID found"}

        try:
            # Import here to avoid circular imports
            from ..agents.orchestrator import run_orchestration

            # Run the orchestrator
            result = run_orchestration(session.data.application_id)

            # Generate summary message
            final_state = result.get("state", {})
            decision = final_state.get("eligibility", {}).get("decision", "UNKNOWN")
            confidence = final_state.get("eligibility", {}).get("confidence", 0)

            if decision == "APPROVE":
                summary_msg = (
                    f"🎉 **Congratulations! Your application has been APPROVED!**\n\n"
                    f"**Decision Confidence:** {confidence:.1%}\n\n"
                    "**Next Steps:**\n"
                    "• You will receive payment details via SMS\n"
                    "• First payment will arrive within 10 business days\n"
                    "• Please provide quarterly updates on any changes\n\n"
                    "Thank you for using our AI-powered application system!"
                )
            else:
                summary_msg = (
                    f"❌ **Your application has been DECLINED**\n\n"
                    f"**Decision Confidence:** {confidence:.1%}\n\n"
                    "**You may appeal this decision within 30 days by:**\n"
                    "• Providing additional documentation\n"
                    "• Scheduling an interview with a case worker\n"
                    "• Submitting an appeal form\n\n"
                    "Thank you for using our service."
                )

            self._add_message(session_id, "assistant", summary_msg, "system")
            session.state = ConversationState.COMPLETED

            return {
                "success": True,
                "decision": decision,
                "confidence": confidence,
                "summary": summary_msg,
                "full_result": result
            }

        except Exception as e:
            error_msg = f"❌ Application processing failed: {str(e)}"
            self._add_message(session_id, "assistant", error_msg, "system")
            return {"error": str(e)}


# Global chat service instance
_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service