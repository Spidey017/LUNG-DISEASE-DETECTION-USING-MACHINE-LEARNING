#!/usr/bin/env python3
"""Remove <b> tags from PDF report"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove <b> tags - keep just the text
replacements = [
    ('result_text = f"<b>✓ {disease_name} DETECTED</b>"', 'result_text = f"✓ {disease_name} DETECTED"'),
    ('result_text = f"<b>✓ NO ABNORMALITIES DETECTED</b>"', 'result_text = f"✓ NO ABNORMALITIES DETECTED"'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Done! Removed all <b> tags from PDF report')
print('Now displays: ✓ PNEUMONIA DETECTED (plain text)')
