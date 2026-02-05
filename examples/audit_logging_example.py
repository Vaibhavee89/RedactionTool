#!/usr/bin/env python3
"""
Example: Enterprise Audit Logging Usage

Demonstrates all audit logging features:
1. Hashed document IDs
2. No raw PII storage
3. Comprehensive audit trail
4. JSON and CSV export
5. Retention management
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.audit import AuditLogger, DocumentHasher, RetentionManager


def example_basic_audit_logging():
    """Example 1: Basic audit logging"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Audit Logging")
    print("=" * 70)

    # Initialize audit logger
    logger = AuditLogger(
        log_dir="audit_logs",
        enable_csv=True,
        retention_days=90,
        hash_documents=True  # Enable privacy protection
    )

    # Simulate redaction event
    entities_detected = [
        {
            "entity_type": "EMAIL",
            "text": "john.doe@company.com",
            "confidence": 0.95,
            "start": 45,
            "end": 65,
            "source": "presidio"
        },
        {
            "entity_type": "PERSON",
            "text": "John Doe",
            "confidence": 0.92,
            "start": 10,
            "end": 18,
            "source": "spacy"
        },
        {
            "entity_type": "PHONE",
            "text": "+1-555-1234",
            "confidence": 0.88,
            "start": 70,
            "end": 81,
            "source": "presidio"
        }
    ]

    actions_taken = {
        "EMAIL": "block",
        "PERSON": "mask",
        "PHONE": "label"
    }

    # Log the redaction event (NO RAW PII STORED)
    log_id = logger.log_redaction_event(
        document_path="/data/hr/employee_records.pdf",
        policy_name="gdpr_basic",
        entities_detected=entities_detected,
        actions_taken=actions_taken,
        success=True,
        processing_time_ms=125.3
    )

    print(f"✓ Logged redaction event: {log_id}")
    print(f"✓ Document ID: HASHED (original path not stored)")
    print(f"✓ Entity text: HASHED (no raw PII in logs)")
    print()

    # Save audit logs
    saved_files = logger.save_session_logs(format="both")
    print("✓ Audit logs saved:")
    print(f"  - JSON: {saved_files['json']}")
    print(f"  - CSV:  {saved_files['csv']}")
    print()

    # Get session summary
    summary = logger.get_session_summary()
    print("✓ Session Summary:")
    print(f"  - Total events logged: {summary['total_logs']}")
    print(f"  - Success rate: {summary['success_rate']:.1f}%")
    print(f"  - Total entities detected: {summary['total_entities_detected']}")
    print(f"  - Entity types: {list(summary['entity_types'].keys())}")
    print(f"  - Privacy protected: {summary['privacy_protected']}")
    print(f"  - Raw PII stored: {summary['raw_pii_stored']}")
    print()


def example_document_hashing():
    """Example 2: Document ID hashing"""
    print("=" * 70)
    print("EXAMPLE 2: Document ID Hashing")
    print("=" * 70)

    hasher = DocumentHasher(use_salt=True)

    # Hash a document path
    doc_path = "/home/user/confidential/financial_report_2024.pdf"
    doc_hash = hasher.hash_document_id(doc_path)

    print(f"Original path: {doc_path}")
    print(f"Hashed ID:     {doc_hash}")
    print()

    # Mask the filepath for logging
    masked_path = hasher.mask_filepath(doc_path, show_last_n=1)
    print(f"Masked path:   {masked_path}")
    print()

    # Get safe document metadata
    metadata = hasher.get_document_metadata(doc_path)
    print("Safe metadata for audit log:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    # Hash entity text (for tracking without storing PII)
    entity_text = "rajesh.kumar@secret.com"
    entity_hash = hasher.hash_entity_text(entity_text)
    print(f"Entity text (PII): {entity_text}")
    print(f"Entity hash:       {entity_hash}")
    print(f"Reversible?        No (one-way hash)")
    print()


def example_batch_processing_with_audit():
    """Example 3: Batch processing with audit logging"""
    print("=" * 70)
    print("EXAMPLE 3: Batch Processing with Audit Trail")
    print("=" * 70)

    logger = AuditLogger(log_dir="audit_logs/batch")

    # Simulate processing multiple documents
    documents = [
        "contract_001.pdf",
        "invoice_002.docx",
        "customer_data.xlsx"
    ]

    for i, doc_name in enumerate(documents):
        # Simulate entity detection
        entities = [
            {"entity_type": "EMAIL", "text": f"user{i}@test.com", "confidence": 0.95, "start": 0, "end": 20, "source": "presidio"}
        ]

        logger.log_redaction_event(
            document_path=f"/batch_input/{doc_name}",
            policy_name="company_policy",
            entities_detected=entities,
            actions_taken={"EMAIL": "block"},
            success=True,
            processing_time_ms=100.0 + i * 10
        )

    print(f"✓ Processed {len(documents)} documents")
    print()

    # Save batch logs
    saved = logger.save_session_logs(session_name="batch_20240205", format="both")
    print("✓ Batch audit logs saved:")
    for format_type, path in saved.items():
        print(f"  - {format_type.upper()}: {path}")
    print()

    # Session summary
    summary = logger.get_session_summary()
    print(f"✓ Batch summary:")
    print(f"  - Documents processed: {summary['total_logs']}")
    print(f"  - Total entities found: {summary['total_entities_detected']}")
    print(f"  - Success rate: {summary['success_rate']:.1f}%")
    print()


def example_retention_management():
    """Example 4: Audit log retention"""
    print("=" * 70)
    print("EXAMPLE 4: Retention Management")
    print("=" * 70)

    # Configure retention policy
    retention = RetentionManager(
        retention_days=90,
        archive_enabled=True,
        archive_path="audit_logs/archive"
    )

    print("Retention Policy:")
    config = retention.get_retention_config()
    print(f"  - Retention period: {config['retention_days']} days")
    print(f"  - Archive enabled: {config['archive_enabled']}")
    print(f"  - Archive location: {config['archive_path']}")
    print(f"  - Cutoff date: {config['cutoff_date']}")
    print()

    # Get retention report
    logger = AuditLogger(log_dir="audit_logs", retention_days=90)
    report = logger.get_retention_report()

    print("Retention Report:")
    print(f"  Active logs:")
    print(f"    - Count: {report['logs']['active']}")
    print(f"    - Size: {report['logs']['total_size_bytes'] / 1024:.1f} KB")
    print(f"  Expired logs:")
    print(f"    - Count: {report['logs']['expired']}")
    print(f"  Archived logs:")
    print(f"    - Count: {report['archived']['total']}")
    print(f"    - Size: {report['archived']['total_size_bytes'] / 1024:.1f} KB")
    print()

    # Clean old logs
    print("Cleaning old logs...")
    stats = logger.clean_old_logs()
    print(f"  - Archived: {stats['archived']} files")
    print(f"  - Cleaned: {stats['cleaned']} files")
    print(f"  - Errors: {stats['errors']} files")
    print()


def example_compliance_workflow():
    """Example 5: Complete compliance workflow"""
    print("=" * 70)
    print("EXAMPLE 5: Complete Compliance Workflow")
    print("=" * 70)

    # 1. Initialize audit logger with enterprise settings
    logger = AuditLogger(
        log_dir="audit_logs/compliance",
        enable_csv=True,
        retention_days=365,  # 1 year retention for compliance
        hash_documents=True
    )

    print("Step 1: Process sensitive document")
    print("----------------------------------------")

    # 2. Process document with redaction
    entities = [
        {"entity_type": "PERSON", "text": "Rajesh Kumar", "confidence": 0.95, "start": 0, "end": 12, "source": "spacy"},
        {"entity_type": "PAN", "text": "ABCDE1234F", "confidence": 0.98, "start": 20, "end": 30, "source": "regex"},
        {"entity_type": "AADHAAR", "text": "1234-5678-9012", "confidence": 0.96, "start": 35, "end": 49, "source": "regex"}
    ]

    actions = {
        "PERSON": "mask",
        "PAN": "block",
        "AADHAAR": "block"
    }

    log_id = logger.log_redaction_event(
        document_path="/compliance/sensitive_financial_data.pdf",
        policy_name="india_finance_compliance",
        entities_detected=entities,
        actions_taken=actions,
        success=True,
        processing_time_ms=156.7
    )

    print(f"✓ Document processed: {log_id}")
    print(f"✓ Entities detected: {len(entities)}")
    print(f"✓ Policy applied: india_finance_compliance")
    print()

    # 3. Save audit trail
    print("Step 2: Generate audit trail")
    print("----------------------------------------")

    saved = logger.save_session_logs(
        session_name="compliance_audit_20240205",
        format="both"
    )

    print("✓ Audit trail generated:")
    print(f"  - JSON report: {saved['json']}")
    print(f"  - CSV report:  {saved['csv']}")
    print()

    # 4. Verify privacy protection
    print("Step 3: Verify privacy protection")
    print("----------------------------------------")

    import json
    with open(saved['json'], 'r') as f:
        audit_data = json.load(f)

    first_log = audit_data['logs'][0]
    print(f"✓ Document ID hashed: {first_log['privacy']['document_id_hashed']}")
    print(f"✓ Entity text hashed: {first_log['privacy']['entity_text_hashed']}")
    print(f"✓ Raw PII stored: {first_log['privacy']['raw_pii_stored']}")
    print(f"✓ Privacy notice: {audit_data['privacy_notice']}")
    print()

    # 5. Compliance summary
    print("Step 4: Compliance summary")
    print("----------------------------------------")

    summary = logger.get_session_summary()
    print("✓ Compliance checklist:")
    print(f"  ☑ Audit trail created: Yes")
    print(f"  ☑ Privacy protected: {summary['privacy_protected']}")
    print(f"  ☑ No PII leakage: {'Yes' if not summary['raw_pii_stored'] else 'No'}")
    print(f"  ☑ Success rate: {summary['success_rate']:.1f}%")
    print(f"  ☑ Retention policy: 365 days")
    print(f"  ☑ Audit format: JSON + CSV")
    print()

    print("✓ Compliance workflow complete!")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("*" * 70)
    print("ENTERPRISE AUDIT LOGGING - USAGE EXAMPLES")
    print("*" * 70)
    print("\n")

    example_basic_audit_logging()
    example_document_hashing()
    example_batch_processing_with_audit()
    example_retention_management()
    example_compliance_workflow()

    print("=" * 70)
    print("🔒 PRIVACY PROTECTION FEATURES DEMONSTRATED:")
    print("=" * 70)
    print("  ✓ Document IDs are hashed (SHA-256)")
    print("  ✓ Entity text is hashed (not stored in plain text)")
    print("  ✓ File paths are masked")
    print("  ✓ No raw PII in audit logs")
    print("  ✓ Comprehensive audit trail")
    print("  ✓ JSON and CSV export")
    print("  ✓ Configurable retention policies")
    print("  ✓ Automatic archival of old logs")
    print()
    print("Ready for enterprise compliance! 🚀")
    print()


if __name__ == "__main__":
    main()
