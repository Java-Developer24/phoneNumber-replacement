import phonenumbers
import re

def is_phone_number(text: str) -> bool:
    """Validate if a string contains a valid phone number format."""
    # Attempt to use phonenumbers library
    # We might need to handle cases where there's no country code, but phonenumbers library works best with + CC.
    # To catch local numbers, we'll also use a regex for general phone-number-like shapes.

    # Try finding with phonenumbers first
    for match in phonenumbers.PhoneNumberMatcher(text, "US"): # Defaulting to US if no country code, just for matching heuristics
        return True

    # Fallback regex for phone numbers since phonenumbers might be too strict on format
    # This regex catches typical local and international patterns including (), -, spaces, +
    # e.g., +1 (234) 567-8901, 98765-43210, +91 9876543210
    pattern = r'(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}'
    matches = re.findall(pattern, text)
    if matches:
        return True

    return False

def extract_phone_numbers_from_text(full_text: str) -> list:
    """Find all valid phone numbers in a block of text."""
    # Replace line breaks with spaces
    normalized_text = full_text.replace('\n', ' ')

    # Use exact regex required
    pattern = r'\b\d{10}\b'

    # Find all matches and use set to remove duplicates
    matches = re.findall(pattern, normalized_text)
    unique_matches = list(set(matches))

    return unique_matches

def get_phone_bounding_boxes(ocr_result: dict) -> list:
    """
    Given the OCR result with words and bounding boxes,
    detect phone numbers and merge bounding boxes of words that make up the phone number.
    Returns a list of bounding boxes: [{"min_x": x, "max_x": x, "min_y": y, "max_y": y}, ...]
    """
    full_text = ocr_result["text"]
    words_info = ocr_result["words"]

    phone_numbers = extract_phone_numbers_from_text(full_text)

    # If no phone numbers found, return early
    if not phone_numbers:
        return []

    # We need to map the matched phone numbers back to the individual words.
    # A phone number could span multiple words (e.g. "+1", "(555)", "123", "4567").

    boxes = []

    # To do this robustly, we reconstruct the text and keep track of character indices
    # However, vision API words might not map 1:1 to our simple regex strings due to spaces.
    # An easier heuristic: for each phone number string, remove all spaces and punctuation.
    # Then iterate through words, and if a sequence of words (ignoring punct) forms the phone number, group them.

    for phone in phone_numbers:
        clean_phone = re.sub(r'[\s\-()+]', '', phone)

        # Sliding window over words
        for i in range(len(words_info)):
            current_concat = ""
            words_in_match = []

            for j in range(i, len(words_info)):
                # Strip out any non-alphanumeric character for more robust matching
                clean_word = re.sub(r'[^0-9a-zA-Z]', '', words_info[j]["text"])
                if not clean_word:
                    continue

                current_concat += clean_word
                words_in_match.append(words_info[j])

                if current_concat == clean_phone:
                    # Found the sequence! Merge bounding boxes
                    min_x = min([w["min_x"] for w in words_in_match])
                    max_x = max([w["max_x"] for w in words_in_match])
                    min_y = min([w["min_y"] for w in words_in_match])
                    max_y = max([w["max_y"] for w in words_in_match])

                    boxes.append({
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "matched_text": phone
                    })
                    break
                elif not clean_phone.startswith(current_concat):
                    # No longer a match
                    break

    return boxes
