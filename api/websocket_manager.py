#!/usr/bin/env python3
"""
WebSocket Manager for Real-time Dashboard Updates
Handles WebSocket connections, authentication, and real-time data broadcasting
"""

import sys
import os
import json
import asyncio
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from security.auth import AuthManager
from api.core.logging import get_logger


logger = get_logger(__name__)

class ConnectionManager:
    """Manage WebSocket connections and broadcasting"""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.auth_manager = AuthManager()

    async def connect(self, websocket: WebSocket, user_id: str, user_info: Dict[str, Any]):
        """Accept a WebSocket connection and store user info"""
        await websocket.accept()

        self.active_connections[user_id] = {
            'websocket': websocket,
            'user_info': user_info,
            'connected_at': datetime.utcnow(),
            'subscriptions': set(),
            'last_activity': datetime.utcnow()
        }

        # Log connection
        self.auth_manager.log_audit_event(
            user_id=user_id,
            action="websocket_connect",
            resource="dashboard",
            details={"connection_time": datetime.utcnow().isoformat()}
        )

        # Send welcome message
        await self.send_personal_message({
            'type': 'connection_established',
            'message': f'Connected to real-time dashboard',
            'server_time': datetime.utcnow().isoformat(),
            'user_role': user_info.get('role')
        }, user_id)

    async def disconnect(self, user_id: str):
        """Disconnect and clean up user connection"""
        if user_id in self.active_connections:
            # Log disconnection
            self.auth_manager.log_audit_event(
                user_id=user_id,
                action="websocket_disconnect",
                resource="dashboard",
                details={
                    "disconnection_time": datetime.utcnow().isoformat(),
                    "session_duration": (datetime.utcnow() - self.active_connections[user_id]['connected_at']).total_seconds()
                }
            )

            del self.active_connections[user_id]

    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]['websocket']
                await websocket.send_text(json.dumps(message))
                self.active_connections[user_id]['last_activity'] = datetime.utcnow()
            except Exception as e:
                logger.exception("websocket_send_failed", user_id=user_id, error=str(e))
                await self.disconnect(user_id)

    async def broadcast_to_role(self, message: Dict[str, Any], allowed_roles: List[str]):
        """Broadcast message to users with specific roles"""
        disconnected_users = []

        for user_id, connection_info in self.active_connections.items():
            user_role = connection_info['user_info'].get('role')

            if user_role in allowed_roles:
                try:
                    websocket = connection_info['websocket']
                    await websocket.send_text(json.dumps(message))
                    connection_info['last_activity'] = datetime.utcnow()
                except Exception as e:
                    logger.exception("websocket_broadcast_failed", user_id=user_id, error=str(e))
                    disconnected_users.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(user_id)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected users"""
        disconnected_users = []

        for user_id, connection_info in self.active_connections.items():
            try:
                websocket = connection_info['websocket']
                await websocket.send_text(json.dumps(message))
                connection_info['last_activity'] = datetime.utcnow()
            except Exception as e:
                logger.exception("websocket_broadcast_failed", user_id=user_id, error=str(e))
                disconnected_users.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(user_id)

    async def subscribe_user(self, user_id: str, subscription_type: str):
        """Subscribe user to specific data updates"""
        if user_id in self.active_connections:
            self.active_connections[user_id]['subscriptions'].add(subscription_type)
            await self.send_personal_message({
                'type': 'subscription_confirmed',
                'subscription': subscription_type
            }, user_id)

    async def unsubscribe_user(self, user_id: str, subscription_type: str):
        """Unsubscribe user from specific data updates"""
        if user_id in self.active_connections:
            self.active_connections[user_id]['subscriptions'].discard(subscription_type)
            await self.send_personal_message({
                'type': 'subscription_removed',
                'subscription': subscription_type
            }, user_id)

    async def send_to_subscribers(self, message: Dict[str, Any], subscription_type: str):
        """Send message to users subscribed to specific updates"""
        disconnected_users = []

        for user_id, connection_info in self.active_connections.items():
            if subscription_type in connection_info['subscriptions']:
                try:
                    websocket = connection_info['websocket']
                    await websocket.send_text(json.dumps(message))
                    connection_info['last_activity'] = datetime.utcnow()
                except Exception as e:
                    logger.exception(
                        "websocket_subscription_send_failed",
                        user_id=user_id,
                        subscription_type=subscription_type,
                        error=str(e),
                    )
                    disconnected_users.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(user_id)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections"""
        total_connections = len(self.active_connections)
        role_counts = {}
        subscription_counts = {}

        for connection_info in self.active_connections.values():
            # Count by role
            role = connection_info['user_info'].get('role', 'UNKNOWN')
            role_counts[role] = role_counts.get(role, 0) + 1

            # Count subscriptions
            for sub in connection_info['subscriptions']:
                subscription_counts[sub] = subscription_counts.get(sub, 0) + 1

        return {
            'total_connections': total_connections,
            'role_distribution': role_counts,
            'subscription_distribution': subscription_counts,
            'server_time': datetime.utcnow().isoformat()
        }

class RealTimeDataService:
    """Service to generate and broadcast real-time updates"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager

    async def broadcast_system_metrics(self):
        """Broadcast system health and performance metrics"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get key metrics
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_people,
                        AVG(cost_hourly) as avg_cost,
                        SUM(fte) as total_fte
                    FROM people
                """)
                people_stats = dict(cursor.fetchone())

                cursor.execute("""
                    SELECT
                        COUNT(*) as total_projects,
                        AVG(cost_of_delay_weekly) as avg_delay_cost
                    FROM projects
                """)
                project_stats = dict(cursor.fetchone())

                cursor.execute("""
                    SELECT COUNT(*) as recent_evidence
                    FROM evidence
                    WHERE date_achieved >= date('now', '-30 days')
                """)
                evidence_stats = dict(cursor.fetchone())

                message = {
                    'type': 'system_metrics',
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': {
                        'team_metrics': people_stats,
                        'project_metrics': project_stats,
                        'evidence_metrics': evidence_stats,
                        'connection_stats': self.connection_manager.get_connection_stats()
                    }
                }

                await self.connection_manager.send_to_subscribers(message, 'system_metrics')

        except Exception as e:
            logger.exception("system_metrics_broadcast_failed", error=str(e))

    async def broadcast_gap_analysis_update(self, project_id: Optional[str] = None):
        """Broadcast gap analysis updates"""
        try:
            # Import gap analysis engine
            from engines.gap_analysis import GapAnalysisEngine
            gap_engine = GapAnalysisEngine()

            if project_id:
                # Project-specific update
                gap_results = gap_engine.analyze_project_gaps(project_id)
                message = {
                    'type': 'gap_analysis_update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'project_id': project_id,
                    'data': gap_results
                }
            else:
                # Portfolio-wide update
                gap_results = gap_engine.analyze_portfolio_gaps()
                message = {
                    'type': 'portfolio_gap_update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': gap_results
                }

            await self.connection_manager.send_to_subscribers(message, 'gap_analysis')

        except Exception as e:
            logger.exception("gap_analysis_broadcast_failed", project_id=project_id, error=str(e))

    async def broadcast_validation_update(self):
        """Broadcast validation framework updates"""
        try:
            from engines.validation import ValidationEngine
            validation_engine = ValidationEngine()

            validation_report = validation_engine.generate_validation_report()
            model_accuracy = validation_engine.run_model_accuracy_tests()

            message = {
                'type': 'validation_update',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'validation_report': validation_report,
                    'model_accuracy': model_accuracy
                }
            }

            await self.connection_manager.send_to_subscribers(message, 'validation')

        except Exception as e:
            logger.exception("validation_broadcast_failed", error=str(e))

    async def broadcast_financial_update(self):
        """Broadcast financial analysis updates"""
        try:
            from engines.financial import FinancialEngine
            financial_engine = FinancialEngine()

            # Get financial summary
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM projects LIMIT 5")
                project_ids = [row["id"] for row in cursor.fetchall()]

            financial_data = {}
            for project_id in project_ids:
                try:
                    recommendations = financial_engine.generate_recommendations(project_id)
                    financial_data[project_id] = recommendations
                except Exception:
                    continue

            message = {
                'type': 'financial_update',
                'timestamp': datetime.utcnow().isoformat(),
                'data': financial_data
            }

            await self.connection_manager.send_to_subscribers(message, 'financial')

        except Exception as e:
            logger.exception("financial_broadcast_failed", error=str(e))

# Global connection manager instance
connection_manager = ConnectionManager()
realtime_service = RealTimeDataService(connection_manager)

# WebSocket router
websocket_router = APIRouter(prefix="/ws", tags=["websocket"])

@websocket_router.websocket("/dashboard")
async def websocket_dashboard(websocket: WebSocket, token: str):
    """WebSocket endpoint for real-time dashboard updates"""
    try:
        # Authenticate user
        auth_manager = AuthManager()
        user_payload = auth_manager.verify_token(token)

        if not user_payload:
            await websocket.close(code=4001, reason="Invalid token")
            return

        user_id = user_payload['user_id']
        await connection_manager.connect(websocket, user_id, user_payload)

        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get('type') == 'subscribe':
                    subscription_type = message.get('subscription')
                    if subscription_type:
                        await connection_manager.subscribe_user(user_id, subscription_type)

                elif message.get('type') == 'unsubscribe':
                    subscription_type = message.get('subscription')
                    if subscription_type:
                        await connection_manager.unsubscribe_user(user_id, subscription_type)

                elif message.get('type') == 'ping':
                    await connection_manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    }, user_id)

                elif message.get('type') == 'request_update':
                    # Client requesting specific data update
                    update_type = message.get('update_type')
                    if update_type == 'system_metrics':
                        await realtime_service.broadcast_system_metrics()
                    elif update_type == 'gap_analysis':
                        project_id = message.get('project_id')
                        await realtime_service.broadcast_gap_analysis_update(project_id)
                    elif update_type == 'validation':
                        await realtime_service.broadcast_validation_update()
                    elif update_type == 'financial':
                        await realtime_service.broadcast_financial_update()

        except WebSocketDisconnect:
            await connection_manager.disconnect(user_id)

    except Exception as e:
        logger.exception("websocket_dashboard_error", error=str(e))
        await websocket.close(code=4000, reason="Server error")

# Background task to send periodic updates
async def periodic_updates():
    """Background task to send periodic updates to connected clients"""
    while True:
        try:
            # Send system metrics every 30 seconds
            await realtime_service.broadcast_system_metrics()
            await asyncio.sleep(30)

            # Send gap analysis update every 5 minutes
            await realtime_service.broadcast_gap_analysis_update()
            await asyncio.sleep(300)

            # Send validation update every 10 minutes
            await realtime_service.broadcast_validation_update()
            await asyncio.sleep(600)

        except Exception as e:
            logger.exception("periodic_update_failed", error=str(e))
            await asyncio.sleep(60)  # Wait before retrying

# Function to start background tasks
def start_background_tasks():
    """Start background tasks for real-time updates"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(periodic_updates())
    except RuntimeError:
        # No event loop running, tasks will start when server starts
        pass