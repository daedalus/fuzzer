#!/usr/bin/env python3
"""
Safely replace random.* calls in operator methods with pool.rng.* calls.
Handles multi-line docstrings correctly.
"""
import re, sys

path = 'src/fuzzer_tool/services/operators.py'
with open(path) as f:
    content = f.read()

# Strategy: parse function-by-function, handling docstrings
# We use a state machine to track if we're inside a docstring

RANDOM_CALL = re.compile(r'(?<!\.)(?<!self\.f\._rand_pool\.)\brandom\.(randint|randrange|choice|random|sample|shuffle)\(')

# Split into lines, track each function's body
lines = content.split('\n')
output = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Detect operator method definitions (with self parameter)
    if re.match(r'^(\s+)def (havoc_mutate|_apply_single_mutation|_op_\w+)\(self', line):
        indent = re.match(r'^(\s+)', line).group(1)
        method_name = re.search(r'def (\w+)', line).group(1)
        
        # Collect method body lines until next method at same level
        body_lines = []
        j = i + 1
        while j < len(lines):
            if j > i + 1 and re.match(rf'^{indent}def \w+\(self', lines[j]):
                break
            if j > i + 1 and lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t') and not lines[j].strip().startswith('#') and lines[j].strip() != '':
                # Check if this looks like a new method (no indent)
                if not lines[j].startswith(indent):
                    break
            body_lines.append(lines[j])
            j += 1
        
        body = '\n'.join(body_lines)
        
        # Check if body has random.* calls
        has_random = bool(RANDOM_CALL.search(body))
        
        if has_random:
            # Find the first executable line (after docstring)
            # Parse docstring: it's the first expression if it starts with """ or '''
            in_doc = False
            doc_end_idx = -1
            for k, bl in enumerate(body_lines):
                stripped = bl.strip()
                if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
                    in_doc = True
                    # Check if docstring is single-line
                    if (stripped.startswith('"""') and stripped.count('"""') >= 2) or \
                       (stripped.startswith("'''") and stripped.count("'''") >= 2):
                        doc_end_idx = k
                        in_doc = False
                    continue
                if in_doc:
                    if stripped.endswith('"""') or stripped.endswith("'''"):
                        # Check for triple quote ending
                        if '"""' in stripped or "'''" in stripped:
                            doc_end_idx = k
                            in_doc = False
                        else:
                            doc_end_idx = k
                            in_doc = False
                    continue
                if stripped and not stripped.startswith('#') and not in_doc:
                    break
            
            # Insert rng = self.f._rand_pool after docstring
            insert_pos = max(doc_end_idx + 1, 0) if doc_end_idx >= 0 else 0
            
            # Check if rng already exists in body
            has_rng = 'rng = ' in body
            has_self_f = 'self.f' in body
            
            if not has_rng:
                rng_line = f'{indent}    rng = self.f._rand_pool'
                body_lines.insert(insert_pos, rng_line)
            
            # Replace random.X(...) with rng.X(...) in ALL body lines
            new_body = []
            for k, bl in enumerate(body_lines):
                new_line = RANDOM_CALL.sub(r'rng.\1(', bl)
                new_body.append(new_line)
            body_lines = new_body
            body = '\n'.join(body_lines)
        
        # Output
        output.append(line)
        output.extend(body_lines)
        i = j
        continue
    
    output.append(line)
    i += 1

# Write result
result = '\n'.join(output)
with open(path, 'w') as f:
    f.write(result)

remaining = len(RANDOM_CALL.findall(result))
print(f"Done. Remaining random.* calls in operators.py: {remaining}")

# Syntax check
import py_compile, tempfile, os
try:
    py_compile.compile(path, doraise=True)
    print("Syntax: OK")
except py_compile.PyCompileError as e:
    print(f"Syntax ERROR: {e}")
