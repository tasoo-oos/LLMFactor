import re

def extract_factors(text: str) -> str:
    pattern = r'\d+\.[^:]+:\s+([^\n]+)'
    matches = re.finditer(pattern, text)
    result_str = ""
    for match in matches:
        result_str += match.group(0) + "\n"
    return result_str[:-1]