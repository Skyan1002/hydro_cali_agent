
from hydrocalib.agents.utils import extract_json_block

# Test cases
valid_json = '{"candidates": [{"id": 1}]}'
malformed_quotes = "{'candidates': [{'id': 1}]}" # Python dict style
markdown_json = '```json\n{"candidates": [{"id": 1}]}\n```'
text_wrapped = 'Here is the json: {"candidates": [{"id": 1}]} thanks.'

print("--- Testing Valid JSON ---")
try:
    print(extract_json_block(valid_json))
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Testing Malformed Quotes (Single Quotes) ---")
try:
    print(extract_json_block(malformed_quotes))
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Testing Markdown JSON ---")
try:
    print(extract_json_block(markdown_json))
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Testing Text Wrapped ---")
try:
    print(extract_json_block(text_wrapped))
except Exception as e:
    print(f"Failed: {e}")
