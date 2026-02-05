"""
Evaluation Dataset Generator.

Generates synthetic and semi-real labeled samples for evaluation.
"""

import json
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os


class EvaluationDatasetGenerator:
    """
    Generate labeled evaluation datasets for PII detection.

    Generates:
    - Synthetic samples (200-500)
    - Semi-real samples
    - Edge cases (boundary conditions, special formats)
    - Multilingual samples (Hindi, Hinglish)
    - Negative samples (no PII)
    """

    # Sample data pools
    INDIAN_NAMES = [
        "Rajesh Kumar Sharma", "Priya Singh", "Amit Patel", "Sunita Gupta",
        "Vikram Reddy", "Anjali Mehta", "Deepak Verma", "Kavita Joshi",
        "Sanjay Rao", "Neha Kapoor", "Manoj Kumar", "Pooja Shah",
        "Anil Agarwal", "Ritu Malhotra", "Suresh Nair", "Geeta Desai"
    ]

    HINDI_NAMES = [
        "राजेश कुमार शर्मा", "प्रिया सिंह", "अमित पटेल", "सुनीता गुप्ता",
        "विक्रम रेड्डी", "अंजलि मेहता", "दीपक वर्मा", "कविता जोशी"
    ]

    CITIES = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore"
    ]

    HINDI_CITIES = [
        "मुंबई", "दिल्ली", "बेंगलुरु", "हैदराबाद", "चेन्नई", "कोलकाता"
    ]

    STREETS = [
        "MG Road", "Park Street", "Brigade Road", "Nehru Place",
        "Gandhi Nagar", "Station Road", "Main Street", "Market Road"
    ]

    EMAIL_DOMAINS = [
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "example.com", "test.com", "company.in", "email.com"
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize dataset generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.samples = []

    def generate_pan(self) -> str:
        """Generate random PAN number."""
        letters1 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        letter2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        letter3 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        digits = ''.join(random.choices('0123456789', k=4))
        letter4 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return f"{letters1}{letter2}{letter3}{digits}{letter4}"

    def generate_aadhaar(self) -> str:
        """Generate random Aadhaar number."""
        parts = [
            ''.join(random.choices('0123456789', k=4)),
            ''.join(random.choices('0123456789', k=4)),
            ''.join(random.choices('0123456789', k=4))
        ]
        return ' '.join(parts)

    def generate_phone(self) -> str:
        """Generate random Indian phone number."""
        prefix = random.choice(['+91-', '91-', ''])
        number = ''.join(random.choices('6789', k=1)) + ''.join(random.choices('0123456789', k=9))
        return f"{prefix}{number}"

    def generate_email(self, name: Optional[str] = None) -> str:
        """Generate random email."""
        if name:
            local = name.lower().replace(' ', '.').split('.')[0] + random.choice(['', str(random.randint(1, 99))])
        else:
            local = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))
        domain = random.choice(self.EMAIL_DOMAINS)
        return f"{local}@{domain}"

    def generate_address(self, language: str = 'en') -> str:
        """Generate random address."""
        if language == 'hi':
            number = random.randint(1, 999)
            street = random.choice(self.STREETS)
            city = random.choice(self.HINDI_CITIES)
            pincode = random.randint(100000, 999999)
            return f"{number} {street}, {city} {pincode}"
        else:
            number = random.randint(1, 999)
            street = random.choice(self.STREETS)
            city = random.choice(self.CITIES)
            state = random.choice(["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu"])
            pincode = random.randint(100000, 999999)
            return f"{number} {street}, {city}, {state} {pincode}"

    def generate_sample(
        self,
        template_type: str = 'basic',
        language: str = 'en',
        include_edge_cases: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single labeled sample.

        Args:
            template_type: Type of template ('basic', 'form', 'letter', 'mixed')
            language: Language ('en', 'hi', 'hinglish')
            include_edge_cases: Include boundary conditions

        Returns:
            Dictionary with text and labeled entities
        """
        if language == 'en':
            name = random.choice(self.INDIAN_NAMES)
        elif language == 'hi':
            name = random.choice(self.HINDI_NAMES)
        else:  # hinglish
            name = random.choice(self.INDIAN_NAMES)

        pan = self.generate_pan()
        aadhaar = self.generate_aadhaar()
        phone = self.generate_phone()
        email = self.generate_email(name)
        address = self.generate_address(language)

        if template_type == 'basic':
            if language == 'hi':
                text = f"नाम: {name}\nपैन: {pan}\nआधार: {aadhaar}\nफोन: {phone}\nईमेल: {email}"
            elif language == 'hinglish':
                text = f"Mera naam hai {name} aur PAN {pan} hai. Phone: {phone}"
            else:
                text = f"Name: {name}\nPAN: {pan}\nAadhaar: {aadhaar}\nPhone: {phone}\nEmail: {email}"

        elif template_type == 'form':
            text = f"""
Personal Information Form

Full Name: {name}
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {phone}
Email Address: {email}
Residential Address: {address}
"""

        elif template_type == 'letter':
            text = f"""
Dear Sir/Madam,

I am {name}, and I would like to apply for the service. My contact details are as follows:

PAN: {pan}
Phone: {phone}
Email: {email}

Please reach out to me at your earliest convenience.

Regards,
{name}
"""

        elif template_type == 'mixed':
            text = f"""
Application Details:
Applicant: {name} (PAN: {pan})
Contact: {phone} | {email}
Address: {address}
Aadhaar Linked: Yes ({aadhaar})
"""

        else:  # default
            text = f"Customer {name}, PAN {pan}, Contact: {phone}"

        # Label entities
        entities = []

        # Find and label all entities in text
        for entity_text, entity_type in [
            (name, 'PERSON' if language == 'en' else 'HINDI_NAME'),
            (pan, 'PAN' if language == 'en' else 'HINDI_PAN'),
            (aadhaar, 'AADHAAR' if language == 'en' else 'HINDI_AADHAAR'),
            (phone, 'PHONE' if language == 'en' else 'HINDI_PHONE'),
            (email, 'EMAIL' if language == 'en' else 'HINDI_EMAIL'),
        ]:
            start = text.find(entity_text)
            if start != -1:
                entities.append({
                    'text': entity_text,
                    'entity_type': entity_type,
                    'start': start,
                    'end': start + len(entity_text),
                    'ground_truth': True
                })

        # Find address mentions
        if address in text:
            start = text.find(address)
            entities.append({
                'text': address,
                'entity_type': 'ADDRESS' if language == 'en' else 'HINDI_ADDRESS',
                'start': start,
                'end': start + len(address),
                'ground_truth': True
            })

        return {
            'id': f"sample_{len(self.samples) + 1}",
            'text': text,
            'entities': entities,
            'language': language,
            'template_type': template_type,
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'is_synthetic': True,
                'has_edge_cases': include_edge_cases
            }
        }

    def generate_negative_sample(self) -> Dict[str, Any]:
        """Generate sample with no PII (negative sample)."""
        templates = [
            "The meeting is scheduled for tomorrow at 10 AM in the conference room.",
            "Please submit the report by end of day. Thank you.",
            "The weather forecast predicts rain this weekend.",
            "Our office is located in the business district near the metro station.",
            "The project deadline has been extended by two weeks.",
            "मीटिंग कल सुबह 10 बजे है। कृपया समय पर आएं।",
            "रिपोर्ट आज शाम तक जमा करें। धन्यवाद।"
        ]

        text = random.choice(templates)

        return {
            'id': f"negative_sample_{len(self.samples) + 1}",
            'text': text,
            'entities': [],  # No entities
            'language': 'en' if text[0].isascii() else 'hi',
            'template_type': 'negative',
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'is_synthetic': True,
                'is_negative': True
            }
        }

    def generate_edge_case_sample(self) -> Dict[str, Any]:
        """Generate samples with edge cases."""
        edge_cases = [
            # Boundary cases
            {
                'text': "PAN:ABCDE1234F Phone:9876543210",  # No spaces
                'entities': [
                    {'text': 'ABCDE1234F', 'entity_type': 'PAN', 'start': 4, 'end': 14, 'ground_truth': True},
                    {'text': '9876543210', 'entity_type': 'PHONE', 'start': 21, 'end': 31, 'ground_truth': True}
                ]
            },
            # Multiple occurrences
            {
                'text': "PAN ABCDE1234F belongs to person with PAN ABCDE1234F",
                'entities': [
                    {'text': 'ABCDE1234F', 'entity_type': 'PAN', 'start': 4, 'end': 14, 'ground_truth': True},
                    {'text': 'ABCDE1234F', 'entity_type': 'PAN', 'start': 43, 'end': 53, 'ground_truth': True}
                ]
            },
            # Mixed language
            {
                'text': "Name is राजेश and PAN is ABCDE1234F",
                'entities': [
                    {'text': 'राजेश', 'entity_type': 'PERSON', 'start': 8, 'end': 13, 'ground_truth': True},
                    {'text': 'ABCDE1234F', 'entity_type': 'PAN', 'start': 26, 'end': 36, 'ground_truth': True}
                ]
            },
            # Special formatting
            {
                'text': "PAN: ABCDE-1234-F",  # Unusual format
                'entities': [
                    {'text': 'ABCDE-1234-F', 'entity_type': 'PAN', 'start': 5, 'end': 17, 'ground_truth': True}
                ]
            }
        ]

        sample = random.choice(edge_cases)
        sample['id'] = f"edge_case_{len(self.samples) + 1}"
        sample['language'] = 'mixed'
        sample['template_type'] = 'edge_case'
        sample['created_at'] = datetime.now().isoformat()
        sample['metadata'] = {'is_synthetic': True, 'is_edge_case': True}

        return sample

    def generate_dataset(
        self,
        num_samples: int = 200,
        language_distribution: Dict[str, float] = None,
        include_negatives: bool = True,
        include_edge_cases: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate complete evaluation dataset.

        Args:
            num_samples: Total number of samples to generate
            language_distribution: Distribution of languages {'en': 0.5, 'hi': 0.3, 'hinglish': 0.2}
            include_negatives: Include negative samples (no PII)
            include_edge_cases: Include edge cases

        Returns:
            List of labeled samples
        """
        if language_distribution is None:
            language_distribution = {'en': 0.6, 'hi': 0.25, 'hinglish': 0.15}

        templates = ['basic', 'form', 'letter', 'mixed']
        self.samples = []

        # Calculate splits
        num_negatives = int(num_samples * 0.1) if include_negatives else 0
        num_edge_cases = int(num_samples * 0.1) if include_edge_cases else 0
        num_regular = num_samples - num_negatives - num_edge_cases

        # Generate regular samples
        for i in range(num_regular):
            # Determine language
            rand = random.random()
            cumulative = 0
            language = 'en'
            for lang, prob in language_distribution.items():
                cumulative += prob
                if rand < cumulative:
                    language = lang
                    break

            template = random.choice(templates)
            sample = self.generate_sample(template, language)
            self.samples.append(sample)

        # Generate negative samples
        for i in range(num_negatives):
            sample = self.generate_negative_sample()
            self.samples.append(sample)

        # Generate edge cases
        for i in range(num_edge_cases):
            sample = self.generate_edge_case_sample()
            self.samples.append(sample)

        # Shuffle
        random.shuffle(self.samples)

        # Assign sequential IDs
        for i, sample in enumerate(self.samples):
            sample['id'] = f"sample_{i+1:04d}"

        return self.samples

    def save_dataset(self, filepath: str):
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'total_samples': len(self.samples),
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'samples': self.samples
            }, f, indent=2, ensure_ascii=False)

    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = data.get('samples', [])
        return self.samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.samples:
            return {}

        total = len(self.samples)
        entity_counts = {}
        language_counts = {}
        template_counts = {}
        negatives = 0
        edge_cases = 0

        for sample in self.samples:
            # Count entities
            for entity in sample.get('entities', []):
                entity_type = entity['entity_type']
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Count languages
            lang = sample.get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1

            # Count templates
            template = sample.get('template_type', 'unknown')
            template_counts[template] = template_counts.get(template, 0) + 1

            # Count special types
            metadata = sample.get('metadata', {})
            if metadata.get('is_negative'):
                negatives += 1
            if metadata.get('is_edge_case'):
                edge_cases += 1

        return {
            'total_samples': total,
            'entity_counts': entity_counts,
            'language_distribution': language_counts,
            'template_distribution': template_counts,
            'negative_samples': negatives,
            'edge_case_samples': edge_cases,
            'average_entities_per_sample': sum(len(s.get('entities', [])) for s in self.samples) / total if total > 0 else 0
        }


# Convenience function
def load_evaluation_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from file."""
    generator = EvaluationDatasetGenerator()
    return generator.load_dataset(filepath)
