"""
Retention Manager - Handle audit log retention policies.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging


class RetentionManager:
    """
    Manages audit log retention policies for compliance.

    Supports:
    - Automatic cleanup of old logs
    - Archival to separate directory
    - Configurable retention periods
    """

    def __init__(
        self,
        retention_days: int = 90,
        archive_enabled: bool = True,
        archive_path: Optional[str] = None
    ):
        """
        Initialize RetentionManager.

        Args:
            retention_days: Number of days to retain logs (default: 90)
            archive_enabled: Whether to archive old logs instead of deleting
            archive_path: Path to archive directory (default: audit_logs/archive/)
        """
        self.retention_days = retention_days
        self.archive_enabled = archive_enabled
        self.archive_path = Path(archive_path) if archive_path else Path("audit_logs/archive")
        self.logger = logging.getLogger(__name__)

        # Create archive directory if needed
        if self.archive_enabled:
            self.archive_path.mkdir(parents=True, exist_ok=True)

    def clean_old_logs(self, logs_dir: str) -> Dict[str, int]:
        """
        Clean up old audit logs based on retention policy.

        Args:
            logs_dir: Directory containing audit logs

        Returns:
            Dictionary with cleanup statistics
        """
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return {"cleaned": 0, "archived": 0, "errors": 0}

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        stats = {"cleaned": 0, "archived": 0, "errors": 0}

        # Find JSON audit log files
        for log_file in logs_path.glob("**/*.json"):
            try:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

                if file_mtime < cutoff_date:
                    if self.archive_enabled:
                        # Archive the file
                        self._archive_log(log_file)
                        stats["archived"] += 1
                    else:
                        # Delete the file
                        log_file.unlink()
                        stats["cleaned"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {log_file}: {e}")
                stats["errors"] += 1

        return stats

    def _archive_log(self, log_file: Path):
        """
        Archive a log file to the archive directory.

        Args:
            log_file: Path to log file
        """
        # Create archive subdirectory based on year/month
        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        archive_subdir = self.archive_path / f"{file_mtime.year}/{file_mtime.month:02d}"
        archive_subdir.mkdir(parents=True, exist_ok=True)

        # Move file to archive
        archive_file = archive_subdir / log_file.name
        shutil.move(str(log_file), str(archive_file))

        self.logger.info(f"Archived: {log_file.name} -> {archive_file}")

    def get_retention_config(self) -> Dict[str, any]:
        """
        Get current retention configuration.

        Returns:
            Dictionary with retention settings
        """
        return {
            "retention_days": self.retention_days,
            "archive_enabled": self.archive_enabled,
            "archive_path": str(self.archive_path),
            "cutoff_date": (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        }

    def list_archived_logs(self) -> List[Dict[str, str]]:
        """
        List all archived audit logs.

        Returns:
            List of archived log metadata
        """
        if not self.archive_path.exists():
            return []

        archived = []
        for log_file in self.archive_path.glob("**/*.json"):
            try:
                stat = log_file.stat()
                archived.append({
                    "filename": log_file.name,
                    "path": str(log_file),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error reading archived log {log_file}: {e}")

        return sorted(archived, key=lambda x: x["modified"], reverse=True)

    def restore_from_archive(self, archive_filename: str, restore_to: str) -> bool:
        """
        Restore a log file from archive.

        Args:
            archive_filename: Name of archived file
            restore_to: Directory to restore to

        Returns:
            True if successful, False otherwise
        """
        # Find the archived file
        archived_file = None
        for log_file in self.archive_path.glob(f"**/{archive_filename}"):
            archived_file = log_file
            break

        if not archived_file:
            self.logger.error(f"Archived file not found: {archive_filename}")
            return False

        try:
            restore_path = Path(restore_to)
            restore_path.mkdir(parents=True, exist_ok=True)

            # Copy (not move) from archive
            restore_file = restore_path / archive_filename
            shutil.copy2(str(archived_file), str(restore_file))

            self.logger.info(f"Restored: {archive_filename} -> {restore_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error restoring {archive_filename}: {e}")
            return False

    def get_retention_report(self, logs_dir: str) -> Dict[str, any]:
        """
        Generate a report on log retention status.

        Args:
            logs_dir: Directory containing audit logs

        Returns:
            Dictionary with retention statistics
        """
        logs_path = Path(logs_dir)
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        report = {
            "retention_policy": self.get_retention_config(),
            "logs": {
                "total": 0,
                "active": 0,
                "expired": 0,
                "total_size_bytes": 0
            },
            "archived": {
                "total": len(self.list_archived_logs()),
                "total_size_bytes": sum(
                    Path(log["path"]).stat().st_size
                    for log in self.list_archived_logs()
                    if Path(log["path"]).exists()
                )
            }
        }

        if logs_path.exists():
            for log_file in logs_path.glob("**/*.json"):
                try:
                    stat = log_file.stat()
                    file_mtime = datetime.fromtimestamp(stat.st_mtime)

                    report["logs"]["total"] += 1
                    report["logs"]["total_size_bytes"] += stat.st_size

                    if file_mtime < cutoff_date:
                        report["logs"]["expired"] += 1
                    else:
                        report["logs"]["active"] += 1

                except Exception:
                    pass

        return report

    @staticmethod
    def load_config(config_file: str) -> 'RetentionManager':
        """
        Load retention configuration from JSON file.

        Args:
            config_file: Path to config file

        Returns:
            RetentionManager instance with loaded config
        """
        config_path = Path(config_file)
        if not config_path.exists():
            # Return default configuration
            return RetentionManager()

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            return RetentionManager(
                retention_days=config.get("retention_days", 90),
                archive_enabled=config.get("archive_enabled", True),
                archive_path=config.get("archive_path")
            )
        except Exception as e:
            logging.error(f"Error loading retention config: {e}")
            return RetentionManager()

    def save_config(self, config_file: str):
        """
        Save retention configuration to JSON file.

        Args:
            config_file: Path to config file
        """
        config = self.get_retention_config()

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
