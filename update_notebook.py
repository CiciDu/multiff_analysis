import json

# Read the notebook
with open('multiff_code/notebooks/planning_analysis/plan_data_exploration.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with scatter_plot_params
for cell in notebook['cells']:
    if 'source' in cell:
        source = ''.join(cell['source'])
        if 'scatter_plot_params = {' in source:
            # Find the line with the closing brace
            lines = cell['source']
            for i, line in enumerate(lines):
                if line.strip() == '}':
                    # Insert the new parameters before the closing brace
                    new_lines = [
                        '    "show_d_curv_to_nxt_ff": True,\n',
                        '    "show_angle_to_nxt_ff": True,\n'
                    ]
                    lines[i:i] = new_lines
                    break
            break

# Write the updated notebook
with open('multiff_code/notebooks/planning_analysis/plan_data_exploration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!")
