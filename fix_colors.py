#!/usr/bin/env python3
"""Fix the PDF report to remove color styling from diagnosis text"""

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the color styling
old_positive = "result_text = f\"<b style='color: red;'>✓ {disease_name} DETECTED</b>\""
new_positive = "result_text = f\"<b>✓ {disease_name} DETECTED</b>\""

old_negative = "result_text = f\"<b style='color: green;'>✓ NO ABNORMALITIES DETECTED</b>\""
new_negative = "result_text = f\"<b>✓ NO ABNORMALITIES DETECTED</b>\""

content = content.replace(old_positive, new_positive)
content = content.replace(old_negative, new_negative)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ File updated successfully! HTML color styling removed from PDF report.')
