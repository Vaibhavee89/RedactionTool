"""
Audit Logger - Enterprise-grade audit logging for compliance.

Features:
- Hashed document IDs (no raw PII in logs)
- Comprehensive audit trail
- JSON and CSV export
- Retention management
"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from .document_hasher import DocumentHasher
from .retention_manager import RetentionManager


class ActionType(Enum):
    """Types of actions that can be audited."""
    DETECT = "detect"
    REDACT = "redact"
    MASK = "mask"
    LABEL = "label"
    SKIP = "skip"
    ERROR = "error"


class AuditLogger:
    """
    Enterprise-grade audit logger with privacy protection.

    NO RAW PII IS STORED IN AUDIT LOGS.
    - Document paths are hashed
    - Entity text is hashed (not stored in plain text)
    - Only metadata and statistics are logged
    """

    def __init__(
        self,
        log_dir: str = "audit_logs",
        enable_csv: bool = True,
        retention_days: int = 90,
        hash_documents: bool = True
    ):
        """
        Initialize AuditLogger.

        Args:
            log_dir: Directory to store audit logs
            enable_csv: Enable CSV export alongside JSON
            retention_days: Number of days to retain logs
            hash_documents: Whether to hash document IDs (recommended: True)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_csv = enable_csv
        self.hash_documents = hash_documents

        # Initialize components
        self.hasher = DocumentHasher(use_salt=True) if hash_documents else None
        self.retention_manager = RetentionManager(
            retention_days=retention_days,
            archive_path=str(self.log_dir / "archive")
        )

        self.logger = logging.getLogger(__name__)

        # Current session logs
        self._session_logs: List[Dict] = []
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_redaction_event(
        self,
        document_path: str,
        policy_name: Optional[str],
        entities_detected: List[Dict],
        actions_taken: Dict[str, str],
        success: bool = True,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ) -> str:
        """
        Log a redaction event with privacy protection.

        Args:
            document_path: Path to document (will be hashed)
            policy_name: Name of policy used (if any)
            entities_detected: List of detected entities (will be anonymized)
            actions_taken: Dictionary of entity_type -> action mappings
            success: Whether redaction was successful
            error_message: Error message if failed
            processing_time_ms: Processing time in milliseconds

        Returns:
            Log entry ID
        """
        # Generate document metadata (hashed)
        if self.hasher:
            doc_metadata = self.hasher.get_document_metadata(
                document_path,
                include_hash=True,
                include_masked_path=True
            )
            document_id = doc_metadata["document_id"]
        else:
            # Fallback: use masked path without hashing
            document_id = Path(document_path).name
            doc_metadata = {
                "filename": Path(document_path).name,
                "extension": Path(document_path).suffix.lower(),
                "masked_path": f"****/{Path(document_path).name}"
            }

        # Anonymize entity data (NO RAW PII)
        anonymized_entities = []
        entity_type_counts = {}

        for entity in entities_detected:
            entity_type = entity.get('entity_type', 'UNKNOWN')

            # Count by type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

            # Store anonymized entity (hash the text, don't store it)
            anonymized_entity = {
                "entity_type": entity_type,
                "entity_hash": self.hasher.hash_entity_text(entity.get('text', '')) if self.hasher else None,
                "confidence": entity.get('confidence', 0.0),
                "source": entity.get('source', 'unknown'),
                "start_position": entity.get('start', -1),
                "end_position": entity.get('end', -1),
                "action_taken": actions_taken.get(entity_type, 'unknown')
            }
            anonymized_entities.append(anonymized_entity)

        # Create log entry
        log_entry = {
            "log_id": f"{self._session_id}_{len(self._session_logs):04d}",
            "timestamp": datetime.now().isoformat(),
            "document": doc_metadata,
            "document_id": document_id,  # Hashed ID (no raw path)
            "policy": {
                "policy_name": policy_name or "default",
                "policy_applied": policy_name is not None
            },
            "entities": {
                "total_detected": len(entities_detected),
                "by_type": entity_type_counts,
                "details": anonymized_entities  # NO RAW PII HERE
            },
            "actions": actions_taken,
            "result": {
                "success": success,
                "error_message": error_message,
                "processing_time_ms": processing_time_ms
            },
            "privacy": {
                "document_id_hashed": self.hash_documents,
                "entity_text_hashed": self.hash_documents,
                "raw_pii_stored": False  # ALWAYS FALSE
            }
        }

        # Add to session logs
        self._session_logs.append(log_entry)

        return log_entry["log_id"]

    def save_session_logs(
        self,
        session_name: Optional[str] = None,
        format: str = "both"  # "json", "csv", or "both"
    ) -> Dict[str, str]:
        """
        Save accumulated session logs to disk.

        Args:
            session_name: Custom name for session (default: timestamp)
            format: Export format ("json", "csv", or "both")

        Returns:
            Dictionary with paths to saved files
        """
        if not self._session_logs:
            self.logger.warning("No logs to save")
            return {}

        session_name = session_name or self._session_id
        saved_files = {}

        # Save as JSON
        if format in ["json", "both"]:
            json_path = self.log_dir / f"audit_{session_name}.json"
            self._save_json(json_path)
            saved_files["json"] = str(json_path)

        # Save as CSV
        if format in ["csv", "both"] and self.enable_csv:
            csv_path = self.log_dir / f"audit_{session_name}.csv"
            self._save_csv(csv_path)
            saved_files["csv"] = str(csv_path)

        self.logger.info(f"Saved {len(self._session_logs)} audit logs to {self.log_dir}")

        return saved_files

    def _save_json(self, filepath: Path):
        """Save logs as JSON."""
        audit_data = {
            "session_id": self._session_id,
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(self._session_logs),
            "privacy_notice": "No raw PII is stored in this audit log. Document IDs and entity text are hashed.",
            "logs": self._session_logs
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

    def _save_csv(self, filepath: Path):
        """Save logs as CSV (flattened structure)."""
        if not self._session_logs:
            return

        # Define CSV columns
        fieldnames = [
            "log_id",
            "timestamp",
            "document_id",
            "filename",
            "extension",
            "masked_path",
            "policy_name",
            "total_entities",
            "entity_types",
            "success",
            "error_message",
            "processing_time_ms"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for log_entry in self._session_logs:
                # Flatten the nested structure
                csv_row = {
                    "log_id": log_entry["log_id"],
                    "timestamp": log_entry["timestamp"],
                    "document_id": log_entry["document_id"],
                    "filename": log_entry["document"]["filename"],
                    "extension": log_entry["document"]["extension"],
                    "masked_path": log_entry["document"].get("masked_path", ""),
                    "policy_name": log_entry["policy"]["policy_name"],
                    "total_entities": log_entry["entities"]["total_detected"],
                    "entity_types": ", ".join(
                        f"{k}:{v}" for k, v in log_entry["entities"]["by_type"].items()
                    ),
                    "success": log_entry["result"]["success"],
                    "error_message": log_entry["result"]["error_message"] or "",
                    "processing_time_ms": log_entry["result"]["processing_time_ms"] or 0
                }
                writer.writerow(csv_row)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.

        Returns:
            Dictionary with session statistics
        """
        if not self._session_logs:
            return {"total_logs": 0}

        total_entities = sum(log["entities"]["total_detected"] for log in self._session_logs)
        successful = sum(1 for log in self._session_logs if log["result"]["success"])
        failed = len(self._session_logs) - successful

        # Entity type breakdown
        all_entity_types = {}
        for log in self._session_logs:
            for entity_type, count in log["entities"]["by_type"].items():
                all_entity_types[entity_type] = all_entity_types.get(entity_type, 0) + count

        # Policy usage
        policy_usage = {}
        for log in self._session_logs:
            policy = log["policy"]["policy_name"]
            policy_usage[policy] = policy_usage.get(policy, 0) + 1

        return {
            "session_id": self._session_id,
            "total_logs": len(self._session_logs),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self._session_logs) * 100,
            "total_entities_detected": total_entities,
            "entity_types": all_entity_types,
            "policies_used": policy_usage,
            "privacy_protected": True,
            "raw_pii_stored": False
        }

    def clear_session(self):
        """Clear current session logs."""
        self._session_logs.clear()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def clean_old_logs(self) -> Dict[str, int]:
        """
        Clean old logs based on retention policy.

        Returns:
            Dictionary with cleanup statistics
        """
        return self.retention_manager.clean_old_logs(str(self.log_dir))

    def get_retention_report(self) -> Dict[str, Any]:
        """
        Get retention policy report.

        Returns:
            Dictionary with retention statistics
        """
        return self.retention_manager.get_retention_report(str(self.log_dir))

    def export_logs(
        self,
        output_path: str,
        format: str = "json",
        date_range: Optional[tuple] = None
    ) -> bool:
        """
        Export audit logs with optional date filtering.

        Args:
            output_path: Path to output file
            format: Export format ("json" or "csv")
            date_range: Optional tuple of (start_date, end_date)

        Returns:
            True if successful
        """
        try:
            # Collect all logs from directory
            all_logs = []
            for log_file in self.log_dir.glob("audit_*.json"):
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    if "logs" in data:
                        all_logs.extend(data["logs"])

            # Filter by date if specified
            if date_range:
                start_date, end_date = date_range
                all_logs = [
                    log for log in all_logs
                    if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
                ]

            # Export in requested format
            output = Path(output_path)
            if format == "json":
                with open(output, 'w') as f:
                    json.dump({"logs": all_logs, "total": len(all_logs)}, f, indent=2)
            elif format == "csv":
                self._export_logs_csv(all_logs, output)

            return True

        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
            return False

    def _export_logs_csv(self, logs: List[Dict], output_path: Path):
        """Export logs to CSV format."""
        if not logs:
            return

        fieldnames = [
            "log_id", "timestamp", "document_id", "filename",
            "policy_name", "total_entities", "success"
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for log in logs:
                writer.writerow({
                    "log_id": log["log_id"],
                    "timestamp": log["timestamp"],
                    "document_id": log["document_id"],
                    "filename": log["document"]["filename"],
                    "policy_name": log["policy"]["policy_name"],
                    "total_entities": log["entities"]["total_detected"],
                    "success": log["result"]["success"]
                })


# Singleton instance for convenience
_audit_logger_instance: Optional[AuditLogger] = None


def get_audit_logger(
    log_dir: str = "audit_logs",
    enable_csv: bool = True,
    retention_days: int = 90
) -> AuditLogger:
    """
    Get singleton AuditLogger instance.

    Args:
        log_dir: Directory to store audit logs
        enable_csv: Enable CSV export
        retention_days: Number of days to retain logs

    Returns:
        AuditLogger instance
    """
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger(
            log_dir=log_dir,
            enable_csv=enable_csv,
            retention_days=retention_days
        )
    return _audit_logger_instance
