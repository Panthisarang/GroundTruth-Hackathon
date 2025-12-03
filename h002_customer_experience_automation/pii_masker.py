import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIType:
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    PERSON = "person_name"

class PIIDetector:
    """Detect and mask Personally Identifiable Information (PII) in text."""
    
    def __init__(self):
        # Compile regex patterns for PII detection
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'
            )
        }
        
        # Common person names (can be extended)
        self.common_names = {
            "john", "smith", "jane", "doe", "michael", "jones", "sarah", "williams",
            "david", "brown", "lisa", "davis", "robert", "miller", "maria", "garcia"
        }
        
        # Dictionary to store mappings between original and masked values
        self.mask_mapping = {}
        self.reverse_mapping = {}
    
    def _generate_mask(self, pii_type: str, value: str) -> str:
        """Generate a consistent mask for a PII value."""
        # Create a key for this PII type and value
        key = f"{pii_type}:{value.lower()}"
        
        # If we've seen this PII before, return the same mask
        if key in self.mask_mapping:
            return self.mask_mapping[key]
        
        # Otherwise, create a new mask
        mask_id = len([k for k in self.mask_mapping.keys() if k.startswith(pii_type)]) + 1
        mask = f"<{pii_type.upper()}_{mask_id}>"
        
        # Store the mapping in both directions
        self.mask_mapping[key] = mask
        self.reverse_mapping[mask] = value
        
        return mask
    
    def _unmask_value(self, masked_value: str) -> str:
        """Convert a masked value back to its original form."""
        return self.reverse_mapping.get(masked_value, masked_value)
    
    def mask_text(self, text: str) -> Tuple[str, Dict[str, List[Dict]]]:
        """Mask PII in the given text.
        
        Args:
            text: Input text potentially containing PII
            
        Returns:
            A tuple of (masked_text, pii_found) where:
            - masked_text: The input text with PII masked
            - pii_found: A dictionary mapping PII types to lists of found values
        """
        if not text or not isinstance(text, str):
            return text, {}
        
        masked_text = text
        pii_found = {pii_type: [] for pii_type in self.patterns}
        
        # First pass: Mask structured PII using regex patterns
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                original = match.group(0)
                mask = self._generate_mask(pii_type, original)
                masked_text = masked_text.replace(original, mask)
                pii_found[pii_type].append({
                    "original": original,
                    "masked": mask,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Second pass: Mask person names (more complex)
        # This is a simple implementation - consider using NER for production
        words = masked_text.split()
        for i, word in enumerate(words):
            # Simple check: title-cased words that aren't at the start of a sentence
            if (word.istitle() and i > 0 and 
                word.lower() in self.common_names and
                not any(m in word for m in self.reverse_mapping)):
                mask = self._generate_mask(PIIType.PERSON, word)
                words[i] = mask
                pii_found.setdefault(PIIType.PERSON, []).append({
                    "original": word,
                    "masked": mask,
                    "start": sum(len(w) + 1 for w in words[:i]),  # Approximate position
                    "end": 0  # Not used for now
                })
        
        masked_text = " ".join(words)
        return masked_text, pii_found
    
    def unmask_text(self, text: str) -> str:
        """Restore original PII values in a masked text.
        
        Args:
            text: Text containing masked PII values
            
        Returns:
            Text with masked values replaced by originals
        """
        if not text:
            return text
            
        # Sort masks by length in descending order to handle nested masks
        masks = sorted(self.reverse_mapping.keys(), key=len, reverse=True)
        
        for mask in masks:
            if mask in text:
                text = text.replace(mask, self.reverse_mapping[mask])
        
        return text

# Create a singleton instance
pii_detector = PIIDetector()

def mask_pii(text: str) -> Tuple[str, Dict]:
    """Convenience function to mask PII in text.
    
    Args:
        text: Input text potentially containing PII
        
    Returns:
        A tuple of (masked_text, pii_found)
    """
    return pii_detector.mask_text(text)

def unmask_pii(text: str) -> str:
    """Convenience function to restore original PII in text.
    
    Args:
        text: Text containing masked PII values
        
    Returns:
        Text with masked values replaced by originals
    """
    return pii_detector.unmask_text(text)

def is_pii(text: str) -> bool:
    """Check if text contains any PII.
    
    Args:
        text: Text to check
        
    Returns:
        True if PII is detected, False otherwise
    """
    if not text:
        return False
        
    masked, pii_found = mask_pii(text)
    return any(len(values) > 0 for values in pii_found.values())
