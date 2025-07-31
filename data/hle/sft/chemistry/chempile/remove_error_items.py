import os
import json
import re


def clean_obj(item):
    if item["answer"].startswith("\\boxed"):
        item["answer"] = item["answer"].replace("\\boxed{", "").replace("}", "")
    item["question"] = item["question"].strip()
    # Remove leading/trailing \n within the <think>...</think> tag in output
    def strip_think_newlines(output):
        def replacer(match):
            content = match.group(1)
            cleaned_content = content.lstrip('\\n').rstrip('\\n')
            return f"<think>{cleaned_content}</think>"
        return re.sub(r"<think>(.*?)</think>", replacer, output, flags=re.DOTALL)
    item["output"] = strip_think_newlines(item["output"])
    return item

with open("chempile_qa_pairs.json", "r") as f:
    data = json.load(f)

new_items = []
for item in data:
    if "error" in item:
        continue
    else:
        item = clean_obj(item)
    new_items.append(item)

with open("chempile_qa_pairs_clean.json", "w") as f:
    json.dump(new_items, f, indent=2, ensure_ascii=False)