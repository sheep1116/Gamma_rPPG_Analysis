const fs = require("fs");
const path = require("path");

const datasets = ["LGI_PPGI", "PURE", "UBFC2"];
const methods = ["CHROM", "GREEN", "LGI", "OMIT", "PBV", "POS"];
const responseTypes = [
  ["positive_sensitive", "Beneficial", "#2A9D8F"],
  ["stable", "Stable", "#577590"],
  ["non_monotonic", "Non-monotonic", "#E9C46A"],
  ["negative_sensitive", "Negative", "#E76F51"],
  ["invalid", "Not evaluable", "#D9D9D9"],
];

function parseCsvLine(line) {
  const fields = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      fields.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  fields.push(value);
  return fields;
}

function loadCounts(file) {
  const lines = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const header = parseCsvLine(lines[0]);
  const methodIndex = header.indexOf("method");
  const typeIndex = header.indexOf("response_type");
  if (methodIndex < 0 || typeIndex < 0) {
    throw new Error(`Required columns are missing from ${file}`);
  }

  const counts = Object.fromEntries(
    methods.map((method) => [
      method,
      Object.fromEntries(responseTypes.map(([key]) => [key, 0])),
    ]),
  );
  for (const line of lines.slice(1)) {
    const fields = parseCsvLine(line);
    const method = fields[methodIndex];
    const responseType = fields[typeIndex];
    if (counts[method] && Object.hasOwn(counts[method], responseType)) {
      counts[method][responseType] += 1;
    }
  }
  return counts;
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function latexFigure(allCounts) {
  const parts = [
    "\\definecolor{responsebeneficial}{HTML}{2A9D8F}",
    "\\definecolor{responsestable}{HTML}{577590}",
    "\\definecolor{responsenonmonotonic}{HTML}{E9C46A}",
    "\\definecolor{responsenegative}{HTML}{E76F51}",
    "\\definecolor{responseinvalid}{HTML}{D9D9D9}",
    "\\begin{tikzpicture}[x=1cm,y=0.038cm,font=\\scriptsize]",
    "  \\draw[fill=responsebeneficial,draw=none] (1.15,119) rectangle (1.45,124);",
    "  \\node[anchor=west] at (1.52,121.5) {Beneficial};",
    "  \\draw[fill=responsestable,draw=none] (3.65,119) rectangle (3.95,124);",
    "  \\node[anchor=west] at (4.02,121.5) {Stable};",
    "  \\draw[fill=responsenonmonotonic,draw=none] (5.45,119) rectangle (5.75,124);",
    "  \\node[anchor=west] at (5.82,121.5) {Non-monotonic};",
    "  \\draw[fill=responsenegative,draw=none] (8.55,119) rectangle (8.85,124);",
    "  \\node[anchor=west] at (8.92,121.5) {Negative};",
    "  \\draw[fill=responseinvalid,draw=none] (10.75,119) rectangle (11.05,124);",
    "  \\node[anchor=west] at (11.12,121.5) {Not evaluable};",
    "  \\node[rotate=90] at (-0.65,50) {Videos (\\%)};",
  ];

  const latexColors = {
    positive_sensitive: "responsebeneficial",
    stable: "responsestable",
    non_monotonic: "responsenonmonotonic",
    negative_sensitive: "responsenegative",
    invalid: "responseinvalid",
  };

  datasets.forEach((dataset, panelIndex) => {
    const counts = allCounts[dataset];
    const panelX = panelIndex * 5.0;
    const barWidth = 0.48;
    const barStep = 0.68;
    parts.push(
      `  \\node[font=\\small\\bfseries] at (${(panelX + 2.15).toFixed(2)},109) {(${String.fromCharCode(97 + panelIndex)}) ${dataset.replace("_", "\\_")}};`,
    );
    for (let tick = 0; tick <= 100; tick += 20) {
      parts.push(
        `  \\draw[gray!35,line width=0.25pt] (${panelX.toFixed(2)},${tick}) -- (${(panelX + 4.30).toFixed(2)},${tick});`,
      );
      if (panelIndex === 0) {
        parts.push(
          `  \\node[anchor=east] at (${(panelX - 0.10).toFixed(2)},${tick}) {${tick}};`,
        );
      }
    }

    methods.forEach((method, methodIndex) => {
      const x = panelX + 0.18 + methodIndex * barStep;
      const total = Object.values(counts[method]).reduce((sum, value) => sum + value, 0);
      let bottom = 0;
      for (const [key] of responseTypes) {
        const top = bottom + (total ? (counts[method][key] / total) * 100 : 0);
        if (top > bottom) {
          parts.push(
            `  \\filldraw[fill=${latexColors[key]},draw=white,line width=0.2pt] (${x.toFixed(3)},${bottom.toFixed(3)}) rectangle (${(x + barWidth).toFixed(3)},${top.toFixed(3)});`,
          );
        }
        bottom = top;
      }
      parts.push(
        `  \\node[anchor=east,rotate=55] at (${(x + barWidth / 2 + 0.08).toFixed(3)},-4) {${method}};`,
      );
    });
    parts.push(
      `  \\draw[black!70,line width=0.4pt] (${panelX.toFixed(2)},0) -- (${(panelX + 4.30).toFixed(2)},0);`,
    );
  });
  parts.push("\\end{tikzpicture}", "");
  return parts.join("\n");
}

function main() {
  const workspace = path.resolve(__dirname, "..", "..", "..");
  const resultRoot = path.join(workspace, "Result", "Result_v5");
  const outputDir = path.join(resultRoot, "paper_figures");
  fs.mkdirSync(outputDir, { recursive: true });

  const width = 1800;
  const height = 620;
  const marginLeft = 92;
  const marginRight = 24;
  const plotTop = 132;
  const plotBottom = 525;
  const panelGap = 36;
  const panelWidth = (width - marginLeft - marginRight - panelGap * 2) / 3;
  const plotHeight = plotBottom - plotTop;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:27px;font-weight:600}.axis{font-size:20px}.tick{font-size:18px}.legend{font-size:19px}</style>',
  ];
  const allCounts = {};

  let legendX = 315;
  for (const [, label, color] of responseTypes) {
    parts.push(`<rect x="${legendX}" y="32" width="28" height="18" rx="2" fill="${color}"/>`);
    parts.push(`<text class="legend" x="${legendX + 38}" y="48">${escapeXml(label)}</text>`);
    legendX += label.length * 10.5 + 72;
  }

  parts.push(
    `<text class="axis" transform="translate(28 ${(plotTop + plotBottom) / 2}) rotate(-90)" text-anchor="middle">Videos (%)</text>`,
  );

  datasets.forEach((dataset, panelIndex) => {
    const file = path.join(
      resultRoot,
      "pre_roi_gamma",
      dataset,
      "pre_roi_sensitivity_labels.csv",
    );
    const counts = loadCounts(file);
    allCounts[dataset] = counts;
    const panelX = marginLeft + panelIndex * (panelWidth + panelGap);
    const barGap = 17;
    const barWidth = (panelWidth - barGap * 7) / 6;

    parts.push(
      `<text class="title" x="${panelX + panelWidth / 2}" y="96" text-anchor="middle">(${String.fromCharCode(97 + panelIndex)}) ${escapeXml(dataset)}</text>`,
    );

    for (let tick = 0; tick <= 100; tick += 20) {
      const y = plotBottom - (tick / 100) * plotHeight;
      parts.push(
        `<line x1="${panelX}" y1="${y}" x2="${panelX + panelWidth}" y2="${y}" stroke="#D8D8D8" stroke-width="1"/>`,
      );
      if (panelIndex === 0) {
        parts.push(
          `<text class="tick" x="${panelX - 12}" y="${y + 6}" text-anchor="end">${tick}</text>`,
        );
      }
    }

    methods.forEach((method, methodIndex) => {
      const x = panelX + barGap + methodIndex * (barWidth + barGap);
      const total = Object.values(counts[method]).reduce((sum, value) => sum + value, 0);
      let bottomY = plotBottom;
      for (const [key, , color] of responseTypes) {
        const percentage = total ? (counts[method][key] / total) * 100 : 0;
        const segmentHeight = (percentage / 100) * plotHeight;
        bottomY -= segmentHeight;
        if (segmentHeight > 0) {
          parts.push(
            `<rect x="${x}" y="${bottomY}" width="${barWidth}" height="${segmentHeight}" fill="${color}" stroke="white" stroke-width="1"/>`,
          );
        }
      }
      const labelX = x + barWidth / 2;
      parts.push(
        `<text class="tick" transform="translate(${labelX - 4} ${plotBottom + 29}) rotate(35)" text-anchor="start">${method}</text>`,
      );
    });

    parts.push(
      `<line x1="${panelX}" y1="${plotBottom}" x2="${panelX + panelWidth}" y2="${plotBottom}" stroke="#333" stroke-width="1.4"/>`,
    );
  });

  parts.push("</svg>");
  const svg = parts.join("\n");
  const svgPath = path.join(outputDir, "pre_roi_response_type_distribution.svg");
  fs.writeFileSync(svgPath, svg, "utf8");
  fs.writeFileSync(
    path.join(outputDir, "pre_roi_response_type_distribution.tex"),
    latexFigure(allCounts),
    "utf8",
  );
}

try {
  main();
} catch (error) {
  console.error(error);
  process.exitCode = 1;
}
