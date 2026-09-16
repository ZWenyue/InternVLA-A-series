#!/usr/bin/env python3
"""Generate clean SVG for Adobe Illustrator without matplotlib."""


def generate_ubca_svg():
    """Create a clean, Illustrator-compatible SVG from scratch."""

    blocks = ["VLM\nPrefix", "4D Future\nExpert", "World Knowledge\nExpert", "Action\nExpert"]
    relations = [
        ["Causal\nself", "Masked", "Masked", "Masked"],
        ["Full\ncross", "Causal\nself", "Causal\nUBCA", "Masked"],
        ["Full\ncross", "Causal\nUBCA", "Causal\nself", "Masked"],
        ["Full\ncross", "Full\ncross", "Full\ncross", "Causal\nself"],
    ]

    colors = {
        "Causal\nself": "#DCEAF7",
        "Full\ncross": "#E8F4E8",
        "Causal\nUBCA": "#FFF0D9",
        "Masked": "#34495E",
    }

    # SVG dimensions and scaling
    cell_size = 100  # pixels per cell
    margin_left = 150
    margin_top = 150
    margin_bottom = 150
    margin_right = 80

    grid_width = 4 * cell_size
    grid_height = 4 * cell_size
    svg_width = margin_left + grid_width + margin_right
    svg_height = margin_top + grid_height + margin_bottom

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '<style>',
        '  text { font-family: Arial, sans-serif; }',
        '  .title { font-size: 24px; font-weight: bold; }',
        '  .label { font-size: 18px; font-weight: bold; }',
        '  .cell-text { font-size: 16px; font-weight: bold; }',
        '  .legend-label { font-size: 14px; }',
        '</style>',
        '<defs>',
        '<style type="text/css">',
        '  * { stroke-linejoin: round; stroke-linecap: butt; }',
        '</style>',
        '</defs>',
        '<g id="background">',
        f'  <rect width="{svg_width}" height="{svg_height}" fill="white"/>',
        '</g>',
    ]

    # Title
    svg_lines.append(f'<text class="title" x="{svg_width/2}" y="40" text-anchor="middle" fill="#263238">Key (attended stream)</text>')

    # Query stream label (left side)
    svg_lines.append(f'<text class="label" x="30" y="{margin_top + grid_height/2}" text-anchor="middle" dominant-baseline="middle" fill="#263238">Query<tspan x="30" dy="1.2em">stream</tspan></text>')

    # Column headers (top)
    for col, label in enumerate(blocks):
        x = margin_left + col * cell_size + cell_size / 2
        y = margin_top - 30
        label_lines = label.split('\n')
        svg_lines.append(f'<text class="label" x="{x}" y="{y}" text-anchor="middle" fill="#263238">{label_lines[0]}<tspan x="{x}" dy="1.2em">{label_lines[1]}</tspan></text>')

    # Draw grid cells
    for row in range(4):
        for col in range(4):
            x = margin_left + col * cell_size
            y = margin_top + row * cell_size
            relation = relations[row][col]
            color = colors[relation]

            # Rectangle
            svg_lines.append(f'  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{color}" stroke="white" stroke-width="3"/>')

            # Text
            text_color = "white" if relation == "Masked" else "#263238"
            text_lines = relation.split('\n')
            text_y = y + cell_size / 2

            # Center text vertically if multiline
            if len(text_lines) == 1:
                svg_lines.append(f'  <text class="cell-text" x="{x + cell_size/2}" y="{text_y}" text-anchor="middle" dominant-baseline="middle" fill="{text_color}">{text_lines[0]}</text>')
            else:
                # Multi-line: position first line
                first_y = text_y - (len(text_lines) - 1) * 8
                for i, line in enumerate(text_lines):
                    svg_lines.append(f'  <text class="cell-text" x="{x + cell_size/2}" y="{first_y + i * 16}" text-anchor="middle" dominant-baseline="middle" fill="{text_color}">{line}</text>')

    # Grid border
    svg_lines.append(f'  <rect x="{margin_left}" y="{margin_top}" width="{grid_width}" height="{grid_height}" fill="none" stroke="#263238" stroke-width="2.5"/>')

    # Row labels (left side)
    for row, label in enumerate(blocks):
        x = margin_left - 20
        y = margin_top + row * cell_size + cell_size / 2
        label_lines = label.split('\n')
        svg_lines.append(f'<text class="label" x="{x}" y="{y}" text-anchor="end" dominant-baseline="middle" fill="#263238">{label_lines[0]}<tspan x="{x}" dy="1.2em">{label_lines[1]}</tspan></text>')

    # Legend
    legend = [
        ("#DCEAF7", "Causal self-attention"),
        ("#E8F4E8", "Full cross-attention"),
        ("#FFF0D9", "Causal UBCA"),
        ("#34495E", "Masked"),
    ]

    legend_start_x = margin_left
    legend_start_y = margin_top + grid_height + 30

    for idx, (color, label_text) in enumerate(legend):
        x = legend_start_x + (idx % 2) * (grid_width / 2)
        y = legend_start_y + (idx // 2) * 25

        # Color box
        svg_lines.append(f'  <rect x="{x}" y="{y}" width="18" height="18" fill="{color}" stroke="#78909C" stroke-width="1"/>')

        # Label
        svg_lines.append(f'  <text class="legend-label" x="{x + 25}" y="{y + 9}" dominant-baseline="middle" fill="#455A64">{label_text}</text>')

    svg_lines.append('</svg>')

    return '\n'.join(svg_lines)


if __name__ == "__main__":
    svg_content = generate_ubca_svg()
    out_path = "/home/l/文档/Research/InternVLA-A-series/4DW-VLA/figures/ubca_mask_clean.svg"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Saved: {out_path}")
