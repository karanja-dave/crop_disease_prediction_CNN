import nbformat
from nbformat import v4 as nbf

# Read the .py script
with open('crop_disease_data_prep.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Create a notebook
nb = nbf.new_notebook()
cells = [nbf.new_code_cell(code)]
nb['cells'] = cells

# Save as .ipynb
with open('crop_disease_data_prep.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook created successfully!")