#!/usr/bin/env python3
import sys

csv_file = r"c:\Users\dhanu\Downloads\AI-300\profiling__automobilepipeline_2026-06-11 08_04_49.csv"

# Read the file
with open(csv_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace subscription ID and tenant ID
original_sub_id = "<SUBSCRIPTION_ID>"
original_tenant_id = "<TENANT_ID>"
placeholder_sub_id = "<SUBSCRIPTION_ID>"
placeholder_tenant_id = "<TENANT_ID>"

content = content.replace(original_sub_id, placeholder_sub_id)
content = content.replace(original_tenant_id, placeholder_tenant_id)

# Write back to file
with open(csv_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully sanitized {csv_file}")
print(f"  - Replaced {original_sub_id} with {placeholder_sub_id}")
print(f"  - Replaced {original_tenant_id} with {placeholder_tenant_id}")
