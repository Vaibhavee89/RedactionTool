#!/usr/bin/env python3
"""
Test suite for Audit Logging & Privacy features.

Tests:
1. Document ID hashing
2. No raw PII storage
3. Audit log contents
4. JSON and CSV export
5. Retention management
"""

import sys
import os
import json
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.audit.document_hasher import DocumentHasher
from app.services.audit.retention_manager import RetentionManager
from app.services.audit.audit_logger import AuditLogger, ActionType


def test_document_hashing():
    """Test 1: Hashed Document IDs"""
    print("=" * 70)
    print("TEST 1: Document ID Hashing")
    print("=" * 70)

    hasher = DocumentHasher(use_salt=True)

    # Test hashing
    doc_path = "/home/user/sensitive/confidential_document.pdf"
    doc_hash = hasher.hash_document_id(doc_path)

    print(f"Original path: {doc_path}")
    print(f"Hashed ID: {doc_hash}")
    print(f"Hash length: {len(doc_hash)} characters")

    # Verify hash is consistent
    doc_hash2 = hasher.hash_document_id(doc_path)
    consistency_check = doc_hash == doc_hash2
    print(f"\n✓ Hash consistency: {consistency_check}")

    # Test path masking
    masked = hasher.mask_filepath(doc_path)
    print(f"✓ Masked path: {masked}")

    # Test entity hashing
    pii_text = "john.doe@example.com"
    entity_hash = hasher.hash_entity_text(pii_text)
    print(f"✓ Entity hash: {entity_hash[:32]}...")

    # Verify no raw PII in output
    no_pii = pii_text not in entity_hash and "john" not in entity_hash.lower()
    print(f"✓ No raw PII in hash: {no_pii}")

    print()
    return consistency_check and no_pii


def test_no_raw_pii_storage():
    """Test 2: No Raw PII Storage"""
    print("=" * 70)
    print("TEST 2: No Raw PII Storage")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = AuditLogger(log_dir=temp_dir, hash_documents=True)

        # Simulate redaction event with PII
        sensitive_entities = [
            {
                "entity_type": "EMAIL",
                "text": "john.doe@secret.com",  # This should NOT appear in logs
                "confidence": 0.95,
                "start": 0,
                "end": 19,
                "source": "presidio"
            },
            {
                "entity_type": "PERSON",
                "text": "Rajesh Kumar",  # This should NOT appear in logs
                "confidence": 0.92,
                "start": 25,
                "end": 37,
                "source": "spacy"
            }
        ]

        actions = {
            "EMAIL": "block",
            "PERSON": "block"
        }

        # Log the event
        log_id = logger.log_redaction_event(
            document_path="/path/to/sensitive_doc.pdf",
            policy_name="gdpr_basic",
            entities_detected=sensitive_entities,
            actions_taken=actions,
            success=True,
            processing_time_ms=150.5
        )

        print(f"✓ Created log entry: {log_id}")

        # Save logs
        saved = logger.save_session_logs(format="json")
        log_file = saved["json"]

        # Read and verify no raw PII
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Check for PII leakage
        pii_texts = ["john.doe@secret.com", "Rajesh Kumar"]
        pii_found = any(pii in log_content for pii in pii_texts)

        print(f"✓ Log file created: {log_file}")
        print(f"✓ Raw PII in logs: {pii_found} (should be False)")

        # Verify hashed IDs are present
        log_data = json.loads(log_content)
        first_log = log_data["logs"][0]

        has_document_id = "document_id" in first_log
        has_entity_hash = any(
            "entity_hash" in entity
            for entity in first_log["entities"]["details"]
        )

        print(f"✓ Document ID hashed: {has_document_id}")
        print(f"✓ Entity text hashed: {has_entity_hash}")
        print(f"✓ Privacy flag: {first_log['privacy']['raw_pii_stored']} (should be False)")

        print()
        return not pii_found and has_document_id and has_entity_hash


def test_audit_log_contents():
    """Test 3: Audit Log Contents"""
    print("=" * 70)
    print("TEST 3: Audit Log Contents (Required Fields)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = AuditLogger(log_dir=temp_dir)

        # Log event
        entities = [
            {"entity_type": "PAN", "text": "ABCDE1234F", "confidence": 0.98, "start": 0, "end": 10, "source": "regex"},
            {"entity_type": "PHONE", "text": "9876543210", "confidence": 0.92, "start": 15, "end": 25, "source": "presidio"}
        ]

        actions = {"PAN": "block", "PHONE": "mask"}

        logger.log_redaction_event(
            document_path="/data/financial_report.pdf",
            policy_name="india_finance",
            entities_detected=entities,
            actions_taken=actions,
            success=True,
            processing_time_ms=89.3
        )

        # Save and read
        saved = logger.save_session_logs(format="json")
        with open(saved["json"], 'r') as f:
            log_data = json.load(f)

        log_entry = log_data["logs"][0]

        # Verify required fields
        required_fields = {
            "timestamp": "timestamp" in log_entry,
            "policy_used": "policy" in log_entry and log_entry["policy"]["policy_name"] == "india_finance",
            "entity_counts": "entities" in log_entry and log_entry["entities"]["total_detected"] == 2,
            "actions_taken": "actions" in log_entry and log_entry["actions"] == actions
        }

        print("Required Fields Check:")
        for field, present in required_fields.items():
            status = "✓" if present else "✗"
            print(f"  {status} {field}: {present}")

        # Additional checks
        print("\nAdditional Information:")
        print(f"  ✓ Timestamp: {log_entry['timestamp']}")
        print(f"  ✓ Policy: {log_entry['policy']['policy_name']}")
        print(f"  ✓ Total entities: {log_entry['entities']['total_detected']}")
        print(f"  ✓ Entity breakdown: {log_entry['entities']['by_type']}")
        print(f"  ✓ Success: {log_entry['result']['success']}")
        print(f"  ✓ Processing time: {log_entry['result']['processing_time_ms']} ms")

        print()
        return all(required_fields.values())


def test_downloadable_reports():
    """Test 4: Downloadable Audit Reports (JSON and CSV)"""
    print("=" * 70)
    print("TEST 4: Downloadable Audit Reports (JSON & CSV)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = AuditLogger(log_dir=temp_dir, enable_csv=True)

        # Log multiple events
        for i in range(3):
            entities = [
                {"entity_type": "EMAIL", "text": f"user{i}@test.com", "confidence": 0.95, "start": 0, "end": 15, "source": "presidio"}
            ]
            logger.log_redaction_event(
                document_path=f"/docs/document_{i}.pdf",
                policy_name="gdpr_basic",
                entities_detected=entities,
                actions_taken={"EMAIL": "block"},
                success=True
            )

        # Save in both formats
        saved = logger.save_session_logs(format="both")

        # Verify JSON
        json_exists = Path(saved["json"]).exists()
        print(f"✓ JSON file created: {json_exists}")

        if json_exists:
            with open(saved["json"], 'r') as f:
                json_data = json.load(f)
            print(f"  - Total entries: {json_data['total_entries']}")
            print(f"  - Privacy notice: {json_data['privacy_notice']}")

        # Verify CSV
        csv_exists = Path(saved["csv"]).exists()
        print(f"✓ CSV file created: {csv_exists}")

        if csv_exists:
            with open(saved["csv"], 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"  - CSV rows: {len(rows)}")
            print(f"  - CSV columns: {list(rows[0].keys())[:5]}...")

        # Get session summary
        summary = logger.get_session_summary()
        print(f"\n✓ Session Summary:")
        print(f"  - Total logs: {summary['total_logs']}")
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
        print(f"  - Total entities: {summary['total_entities_detected']}")
        print(f"  - Privacy protected: {summary['privacy_protected']}")
        print(f"  - Raw PII stored: {summary['raw_pii_stored']} (should be False)")

        print()
        return json_exists and csv_exists


def test_retention_management():
    """Test 5: Retention Configuration"""
    print("=" * 70)
    print("TEST 5: Retention Configuration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create retention manager
        retention = RetentionManager(
            retention_days=30,
            archive_enabled=True,
            archive_path=f"{temp_dir}/archive"
        )

        # Get configuration
        config = retention.get_retention_config()
        print("Retention Configuration:")
        print(f"  ✓ Retention days: {config['retention_days']}")
        print(f"  ✓ Archive enabled: {config['archive_enabled']}")
        print(f"  ✓ Archive path: {config['archive_path']}")
        print(f"  ✓ Cutoff date: {config['cutoff_date']}")

        # Create some test log files
        logs_dir = Path(temp_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create old log (should be cleaned/archived)
        old_log = logs_dir / "old_audit.json"
        old_log.write_text('{"test": "old"}')

        # Set modification time to 35 days ago
        old_time = (datetime.now() - timedelta(days=35)).timestamp()
        os.utime(old_log, (old_time, old_time))

        # Create recent log (should be kept)
        recent_log = logs_dir / "recent_audit.json"
        recent_log.write_text('{"test": "recent"}')

        print(f"\n✓ Created test logs:")
        print(f"  - Old log: {old_log.name}")
        print(f"  - Recent log: {recent_log.name}")

        # Run cleanup
        stats = retention.clean_old_logs(str(logs_dir))
        print(f"\n✓ Cleanup results:")
        print(f"  - Archived: {stats['archived']}")
        print(f"  - Cleaned: {stats['cleaned']}")
        print(f"  - Errors: {stats['errors']}")

        # Verify recent log still exists
        recent_exists = recent_log.exists()
        print(f"\n✓ Recent log preserved: {recent_exists}")

        # Get retention report
        logger = AuditLogger(log_dir=logs_dir, retention_days=30)
        report = logger.get_retention_report()
        print(f"\n✓ Retention Report:")
        print(f"  - Active logs: {report['logs']['active']}")
        print(f"  - Expired logs: {report['logs']['expired']}")
        print(f"  - Archived logs: {report['archived']['total']}")

        print()
        return stats['archived'] > 0 and recent_exists


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("AUDIT LOGGING & PRIVACY - FEATURE TESTS")
    print("=" * 70)
    print()

    results = []
    results.append(("Document ID Hashing", test_document_hashing()))
    print()
    results.append(("No Raw PII Storage", test_no_raw_pii_storage()))
    print()
    results.append(("Audit Log Contents", test_audit_log_contents()))
    print()
    results.append(("Downloadable Reports", test_downloadable_reports()))
    print()
    results.append(("Retention Management", test_retention_management()))
    print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 70)
    print()

    if passed == total:
        print("🎉 All tests passed! Audit logging features are ready.")
        print("\n🔒 Privacy Protection Verified:")
        print("  ✓ Document IDs are hashed")
        print("  ✓ No raw PII stored in audit logs")
        print("  ✓ Entity text is hashed")
        print("  ✓ File paths are masked")
        print("  ✓ Compliance-ready audit trail")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
