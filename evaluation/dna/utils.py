import re

# Reasoning tag extraction templates
REASONING_TAG_TEMPLATES = {
    "think": [
        r"<think>(.*?)</think>",
        r"<thinking>(.*?)</thinking>",
        r"<thoughts>(.*?)</thoughts>"
    ],
    "reasoning": [
        r"<reasoning>(.*?)</reasoning>",
        r"<reason>(.*?)</reason>",
        r"<rationale>(.*?)</rationale>"
    ],
    "analysis": [
        r"<analysis>(.*?)</analysis>",
        r"<analyze>(.*?)</analyze>",
        r"<consideration>(.*?)</consideration>"
    ]
}

def extract_reasoning_content(response: str) -> str:
    """Extract reasoning content from LLM response using tag templates."""
    reasoning_content = ""
    
    for patterns in REASONING_TAG_TEMPLATES.values():
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                reasoning_content += " ".join(matches) + " "
    
    return reasoning_content.strip()


def parse_reasoning_labels(responses: list[str]) -> tuple[list[int], list[int]]:
    """Parse response and reasoning labels from GPT-4 evaluation responses."""
    response_labels = []
    reasoning_labels = []
    
    for response in responses:
        # Parse response label
        response_match = re.search(r'<answer>(\d+)</answer>', response)
        response_label = int(response_match.group(1)) if response_match else -1
        response_labels.append(response_label)
        
        # Parse reasoning label
        reasoning_match = re.search(r'<reasoning>(\d+)</reasoning>', response)
        reasoning_label = int(reasoning_match.group(1)) if reasoning_match else -1
        reasoning_labels.append(reasoning_label)
    
    return response_labels, reasoning_labels
