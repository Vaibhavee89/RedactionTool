"""
Custom Presidio recognizers for Indian government IDs and other entities.
"""

from presidio_analyzer import Pattern, PatternRecognizer
from typing import List, Optional


class PANRecognizer(PatternRecognizer):
    """
    Recognizer for Indian PAN (Permanent Account Number) cards.
    Format: AAAAA9999A
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="pan_pattern",
                regex=r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
                score=0.9
            )
        ]
        super().__init__(
            supported_entity="PAN",
            patterns=patterns,
            context=["pan", "permanent", "account", "number", "tax"]
        )


class AadhaarRecognizer(PatternRecognizer):
    """
    Recognizer for Indian Aadhaar numbers.
    Format: 9999 9999 9999 or 999999999999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="aadhaar_spaced",
                regex=r'\b\d{4}\s\d{4}\s\d{4}\b',
                score=0.9
            ),
            Pattern(
                name="aadhaar_no_space",
                regex=r'\b\d{12}\b',
                score=0.7  # Lower score due to ambiguity
            )
        ]
        super().__init__(
            supported_entity="AADHAAR",
            patterns=patterns,
            context=["aadhaar", "uid", "enrollment", "biometric"]
        )


class VoterIDRecognizer(PatternRecognizer):
    """
    Recognizer for Indian Voter ID cards.
    Format: AAA9999999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="voter_id_pattern",
                regex=r'\b[A-Z]{3}[0-9]{7}\b',
                score=0.85
            )
        ]
        super().__init__(
            supported_entity="VOTER_ID",
            patterns=patterns,
            context=["voter", "election", "epic"]
        )


class IndianDrivingLicenseRecognizer(PatternRecognizer):
    """
    Recognizer for Indian Driving License.
    Format: AA-99-9999-9999999 or AA99999999999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="dl_with_dash",
                regex=r'\b[A-Z]{2}-\d{2}-\d{4}-\d{7}\b',
                score=0.9
            ),
            Pattern(
                name="dl_no_dash",
                regex=r'\b[A-Z]{2}\d{13}\b',
                score=0.85
            )
        ]
        super().__init__(
            supported_entity="DRIVING_LICENSE",
            patterns=patterns,
            context=["driving", "license", "dl", "transport"]
        )


class IFSCRecognizer(PatternRecognizer):
    """
    Recognizer for Indian IFSC (Indian Financial System Code).
    Format: AAAA0999999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="ifsc_pattern",
                regex=r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
                score=0.9
            )
        ]
        super().__init__(
            supported_entity="IFSC_CODE",
            patterns=patterns,
            context=["ifsc", "bank", "code", "branch"]
        )


class PassportRecognizer(PatternRecognizer):
    """
    Recognizer for Indian Passport numbers.
    Format: A9999999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="passport_pattern",
                regex=r'\b[A-Z]\d{7}\b',
                score=0.85
            )
        ]
        super().__init__(
            supported_entity="PASSPORT",
            patterns=patterns,
            context=["passport", "travel", "document"]
        )


class VehicleRegistrationRecognizer(PatternRecognizer):
    """
    Recognizer for Indian Vehicle Registration numbers.
    Format: AA-99-AA-9999 or AA99AA9999
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="vehicle_with_dash",
                regex=r'\b[A-Z]{2}-\d{2}-[A-Z]{1,2}-\d{4}\b',
                score=0.85
            ),
            Pattern(
                name="vehicle_no_dash",
                regex=r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b',
                score=0.80
            )
        ]
        super().__init__(
            supported_entity="VEHICLE_REGISTRATION",
            patterns=patterns,
            context=["vehicle", "registration", "number", "car", "bike"]
        )


class MedicalRecordRecognizer(PatternRecognizer):
    """
    Recognizer for Medical Record Numbers.
    """

    def __init__(self):
        patterns = [
            Pattern(
                name="mrn_pattern",
                regex=r'\bMRN[-\s:]?\d{6,10}\b',
                score=0.85
            ),
            Pattern(
                name="patient_id",
                regex=r'\b(PATIENT|PATIENT[-_]ID|PID)[-\s:]?\d{6,10}\b',
                score=0.80
            )
        ]
        super().__init__(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=patterns,
            context=["medical", "record", "patient", "hospital", "mrn"]
        )


# Note: CreditCardRecognizer removed because Presidio has a built-in one
# that conflicts with custom implementation


def get_indian_recognizers() -> List[PatternRecognizer]:
    """
    Get all Indian-specific recognizers.

    Returns:
        List of custom recognizers
    """
    return [
        PANRecognizer(),
        AadhaarRecognizer(),
        VoterIDRecognizer(),
        IndianDrivingLicenseRecognizer(),
        IFSCRecognizer(),
        PassportRecognizer(),
        VehicleRegistrationRecognizer(),
        MedicalRecordRecognizer()
        # Note: CreditCardRecognizer not included - Presidio has built-in
    ]


def get_all_custom_recognizers() -> List[PatternRecognizer]:
    """
    Get all custom recognizers (Indian + general).

    Returns:
        List of all custom recognizers
    """
    return get_indian_recognizers()
