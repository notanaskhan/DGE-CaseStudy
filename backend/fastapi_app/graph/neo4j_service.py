"""
Neo4j graph service for Phase 5 household graph management.
Handles node creation, relationships, and graph operations.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    # Type placeholder for when neo4j is not available
    class Driver:
        pass

from .entity_extraction import ExtractedEntity
from .entity_normalization import EntityMatch


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    node_id: str
    label: str
    properties: Dict[str, Any]
    created_at: datetime


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph."""
    from_node: str
    to_node: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: datetime


class Neo4jService:
    """Service for managing Neo4j graph operations."""

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")

        # Default to local Neo4j instance
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        self.driver: Optional[Driver] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self._connected = True
                    self._create_constraints()
                    return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False

        return False

    def disconnect(self):
        """Disconnect from Neo4j database."""
        if self.driver:
            self.driver.close()
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected

    def _create_constraints(self):
        """Create database constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT application_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT address_id IF NOT EXISTS FOR (addr:Address) REQUIRE addr.id IS UNIQUE",
            "CREATE CONSTRAINT bank_account_id IF NOT EXISTS FOR (ba:BankAccount) REQUIRE ba.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX person_emirates_id IF NOT EXISTS FOR (p:Person) ON (p.emirates_id)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)",
            "CREATE INDEX application_date IF NOT EXISTS FOR (a:Application) ON (a.submitted_at)",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint might already exist

            for index in indexes:
                try:
                    session.run(index)
                except Exception:
                    pass  # Index might already exist

    def create_person_node(self, entity: ExtractedEntity, application_id: str) -> str:
        """Create or update a person node."""
        person_id = self._generate_person_id(entity.normalized)

        query = """
        MERGE (p:Person {id: $person_id})
        ON CREATE SET
            p.name = $name,
            p.normalized_name = $normalized_name,
            p.created_at = datetime(),
            p.source_applications = [$application_id]
        ON MATCH SET
            p.source_applications = p.source_applications + $application_id,
            p.updated_at = datetime()
        RETURN p.id as node_id
        """

        with self.driver.session() as session:
            result = session.run(query, {
                'person_id': person_id,
                'name': entity.value,
                'normalized_name': entity.normalized,
                'application_id': application_id
            })
            return result.single()["node_id"]

    def create_emirates_id_node(self, entity: ExtractedEntity, person_id: str) -> str:
        """Create Emirates ID node and link to person."""
        emirates_id = entity.normalized

        # Update person with Emirates ID
        query = """
        MATCH (p:Person {id: $person_id})
        SET p.emirates_id = $emirates_id
        RETURN p.id as node_id
        """

        with self.driver.session() as session:
            result = session.run(query, {
                'person_id': person_id,
                'emirates_id': emirates_id
            })
            return result.single()["node_id"]

    def create_address_node(self, entity: ExtractedEntity, application_id: str) -> str:
        """Create or update an address node."""
        address_id = self._generate_address_id(entity.normalized)

        query = """
        MERGE (addr:Address {id: $address_id})
        ON CREATE SET
            addr.address = $address,
            addr.normalized_address = $normalized_address,
            addr.created_at = datetime(),
            addr.source_applications = [$application_id]
        ON MATCH SET
            addr.source_applications = addr.source_applications + $application_id,
            addr.updated_at = datetime()
        RETURN addr.id as node_id
        """

        with self.driver.session() as session:
            result = session.run(query, {
                'address_id': address_id,
                'address': entity.value,
                'normalized_address': entity.normalized,
                'application_id': application_id
            })
            return result.single()["node_id"]

    def create_bank_account_node(self, entity: ExtractedEntity, application_id: str) -> str:
        """Create or update a bank account node."""
        account_id = self._generate_account_id(entity.normalized)

        query = """
        MERGE (ba:BankAccount {id: $account_id})
        ON CREATE SET
            ba.account_number = $account_number,
            ba.iban = $iban,
            ba.created_at = datetime(),
            ba.source_applications = [$application_id]
        ON MATCH SET
            ba.source_applications = ba.source_applications + $application_id,
            ba.updated_at = datetime()
        RETURN ba.id as node_id
        """

        # Determine if it's IBAN or account number
        iban = entity.normalized if entity.entity_type == 'iban' else None
        account_number = entity.normalized if entity.entity_type != 'iban' else None

        with self.driver.session() as session:
            result = session.run(query, {
                'account_id': account_id,
                'account_number': account_number,
                'iban': iban,
                'application_id': application_id
            })
            return result.single()["node_id"]

    def create_application_node(self, application_id: str, application_data: Dict[str, Any]) -> str:
        """Create an application node."""
        app_row = application_data.get('app_row', {})

        query = """
        MERGE (a:Application {id: $application_id})
        ON CREATE SET
            a.submitted_at = datetime($submitted_at),
            a.declared_income = $declared_income,
            a.household_size = $household_size,
            a.channel = $channel,
            a.status = $status,
            a.created_at = datetime()
        RETURN a.id as node_id
        """

        with self.driver.session() as session:
            result = session.run(query, {
                'application_id': application_id,
                'submitted_at': app_row.get('submitted_at', datetime.utcnow().isoformat()),
                'declared_income': app_row.get('declared_monthly_income', 0),
                'household_size': app_row.get('household_size', 1),
                'channel': app_row.get('channel', 'Unknown'),
                'status': app_row.get('status', 'NEW')
            })
            return result.single()["node_id"]

    def create_relationship(self, from_node_id: str, to_node_id: str,
                          relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Create a relationship between two nodes."""
        properties = properties or {}

        query = f"""
        MATCH (from_node), (to_node)
        WHERE from_node.id = $from_id AND to_node.id = $to_id
        MERGE (from_node)-[r:{relationship_type}]->(to_node)
        ON CREATE SET r.created_at = datetime()
        SET r += $properties
        RETURN r
        """

        with self.driver.session() as session:
            result = session.run(query, {
                'from_id': from_node_id,
                'to_id': to_node_id,
                'properties': properties
            })
            return result.single() is not None

    def populate_graph_from_entities(self, application_id: str, entities: List[ExtractedEntity],
                                   application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Populate the graph with entities from an application."""
        results = {
            'application_id': application_id,
            'nodes_created': 0,
            'relationships_created': 0,
            'entities_processed': len(entities)
        }

        if not self.is_connected():
            if not self.connect():
                results['error'] = 'Failed to connect to Neo4j'
                return results

        try:
            # Create application node
            app_node_id = self.create_application_node(application_id, application_data)
            results['nodes_created'] += 1

            # Group entities by type
            person_entities = [e for e in entities if e.entity_type == 'person']
            emirates_id_entities = [e for e in entities if e.entity_type == 'emirates_id']
            address_entities = [e for e in entities if e.entity_type == 'address']
            bank_entities = [e for e in entities if e.entity_type in ['iban', 'bank_account']]

            person_node_ids = []

            # Create person nodes
            for person_entity in person_entities:
                person_id = self.create_person_node(person_entity, application_id)
                person_node_ids.append(person_id)
                results['nodes_created'] += 1

                # Link person to application
                self.create_relationship(person_id, application_id, 'APPLIED_FOR', {
                    'submitted_at': datetime.utcnow().isoformat(),
                    'confidence': person_entity.confidence
                })
                results['relationships_created'] += 1

            # Link Emirates IDs to persons
            for emirates_entity in emirates_id_entities:
                for person_id in person_node_ids:
                    self.create_emirates_id_node(emirates_entity, person_id)

            # Create address nodes and relationships
            for address_entity in address_entities:
                address_id = self.create_address_node(address_entity, application_id)
                results['nodes_created'] += 1

                # Link persons to addresses
                for person_id in person_node_ids:
                    self.create_relationship(person_id, address_id, 'LIVES_AT', {
                        'confidence': address_entity.confidence,
                        'source': address_entity.source
                    })
                    results['relationships_created'] += 1

            # Create bank account nodes and relationships
            for bank_entity in bank_entities:
                account_id = self.create_bank_account_node(bank_entity, application_id)
                results['nodes_created'] += 1

                # Link persons to bank accounts
                for person_id in person_node_ids:
                    self.create_relationship(person_id, account_id, 'OWNS_ACCOUNT', {
                        'confidence': bank_entity.confidence,
                        'source': bank_entity.source
                    })
                    results['relationships_created'] += 1

        except Exception as e:
            results['error'] = str(e)

        return results

    def find_related_applications(self, application_id: str) -> List[Dict[str, Any]]:
        """Find applications related to the given application."""
        if not self.is_connected():
            return []

        query = """
        MATCH (a:Application {id: $application_id})<-[:APPLIED_FOR]-(p:Person)
        MATCH (p)-[:APPLIED_FOR]->(related_app:Application)
        WHERE related_app.id <> $application_id
        RETURN DISTINCT related_app.id as application_id,
               related_app.submitted_at as submitted_at,
               related_app.status as status,
               p.name as person_name
        ORDER BY related_app.submitted_at DESC
        """

        with self.driver.session() as session:
            result = session.run(query, {'application_id': application_id})
            return [dict(record) for record in result]

    def get_household_network(self, application_id: str) -> Dict[str, Any]:
        """Get the household network for an application."""
        if not self.is_connected():
            return {}

        query = """
        MATCH (a:Application {id: $application_id})<-[:APPLIED_FOR]-(p:Person)
        OPTIONAL MATCH (p)-[:LIVES_AT]->(addr:Address)
        OPTIONAL MATCH (p)-[:OWNS_ACCOUNT]->(ba:BankAccount)
        OPTIONAL MATCH (addr)<-[:LIVES_AT]-(household_member:Person)
        WHERE household_member.id <> p.id
        RETURN p.name as applicant_name,
               p.emirates_id as emirates_id,
               addr.address as address,
               collect(DISTINCT household_member.name) as household_members,
               collect(DISTINCT ba.account_number) as bank_accounts
        """

        with self.driver.session() as session:
            result = session.run(query, {'application_id': application_id})
            record = result.single()
            if record:
                return dict(record)
            return {}

    def _generate_person_id(self, normalized_name: str) -> str:
        """Generate a consistent ID for a person."""
        # Use hash of normalized name for consistent IDs
        import hashlib
        return f"person_{hashlib.md5(normalized_name.encode()).hexdigest()[:12]}"

    def _generate_address_id(self, normalized_address: str) -> str:
        """Generate a consistent ID for an address."""
        import hashlib
        return f"addr_{hashlib.md5(normalized_address.encode()).hexdigest()[:12]}"

    def _generate_account_id(self, account_info: str) -> str:
        """Generate a consistent ID for a bank account."""
        import hashlib
        return f"account_{hashlib.md5(account_info.encode()).hexdigest()[:12]}"


# Global service instance
_neo4j_service = None

def get_neo4j_service() -> Neo4jService:
    """Get or create the global Neo4j service instance."""
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service

def ensure_neo4j_connection() -> bool:
    """Ensure Neo4j connection is established."""
    service = get_neo4j_service()
    if not service.is_connected():
        return service.connect()
    return True