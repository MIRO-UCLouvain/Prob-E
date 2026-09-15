import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


# ---------------------------
# Load JSON files
# ---------------------------
def load_results(folder: Path):
    results = {}
    print(f"\n=== Loading folder: {folder} ===")

    for f in folder.glob("*.json"):
        print(f"Loading: {f.name} ... ", end="")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results[f.stem] = {"data": data, "path": f}
            print(f"✅ Loaded: {f.name}")
        except Exception as e:
            print(f"❌ Failed: {f.name} -> {e}")

    return results


# ---------------------------
# Convert JSON -> payload
# ---------------------------
def build_payload(results):
    payload = []

    def format_nominal(goal: str, raw_value):
        """Format a nominal value, rounded to two decimals, with the unit implied by the goal string.

        Vxcc goals ("V60.0Gy<=3.00cc") evaluate to an absolute volume in cc, Vx goals
        ("V50.0Gy>=95.00%") to a fraction of the structure volume, every other goal
        (Dx, Dxcc, Dmean, Dmax, Dmin) to a dose in Gy. The pass/fail colour comes from the
        unrounded value, so a value just below a threshold that rounds onto it still shows as failed.
        """
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)

        goal = goal.strip()
        if goal.endswith("cc"):
            return f"{value:.2f}cc"
        if goal.startswith("V"):
            return f"{value * 100:.2f}%"
        return f"{value:.2f}Gy"

    for key, obj in results.items():
        data = obj["data"]
        path = obj["path"]

        probs = data.get("Probability Array", [])
        table = data.get("Table", [])

        if not table:
            continue

        prob_rows = []
        nominal_rows = []

        for row in table:
            goal = row.get("Clinical Goal", "")
            nominal = format_nominal(goal, row.get("Nominal Value", "N/A"))

            is_probabilistic = bool(row.get("Probabilistic Objective", False))
            success = row.get("Success Array", [])

            # Backward compatibility if the explicit flag is absent.
            if not is_probabilistic and isinstance(success, list) and probs:
                is_probabilistic = len(success) == len(probs)

            if is_probabilistic and isinstance(success, list) and probs and len(success) == len(probs):
                prob_rows.append(
                    {
                        "roi_name": row.get("Mask Name", ""),
                        "clinical_goal": goal,
                        "nominal": nominal,
                        "nominal_passed": row.get("Nominal Success"),
                        "success": success,
                    }
                )
            else:
                nominal_rows.append(
                    {
                        "roi_name": row.get("Mask Name", ""),
                        "clinical_goal": goal,
                        "nominal": nominal,
                        "nominal_passed": row.get("Nominal Success"),
                    }
                )

        payload.append(
            {
                "key": key,
                "file_path": str(path),
                "probability": float(data.get("Probability Mass", sum(probs) if probs else 1.0)),
                "prob_rows": prob_rows,
                "nominal_rows": nominal_rows,
                "weights": probs,
            }
        )

    return payload


# ---------------------------
# HTML UI
# ---------------------------
HTML = """
<html>
<head>
<meta charset="utf-8">
<title>Evaluation Viewer</title>

<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
  background: #f4f6f8;
  margin: 0;
}

.container {
  width: 95%;
  margin: auto;
  padding: 20px 0;
}

.card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  margin-bottom: 24px;
  overflow: hidden;
}

.card-header {
  padding: 12px 16px;
  font-weight: 600;
  background: #4C72B0;
  color: white;
}

.section-title {
  padding: 10px 16px;
  font-weight: 600;
  background: #f2f5fb;
  border-top: 1px solid #e4e9f3;
  border-bottom: 1px solid #e4e9f3;
}

.section-tools {
  float: right;
}

.table-wrap {
  max-height: 70vh;
  overflow: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th {
  position: sticky;
  top: 0;
  background: #e9eef6;
  font-weight: 600;
  padding: 8px;
  text-align: left;
}

td {
  padding: 8px;
  border-bottom: 1px solid #eee;
  font-size: 13px;
}

.num {
  text-align: right;
}

.pass { color: #1b8a3e; font-weight: 600; }
.fail { color: #c62828; font-weight: 600; }

tr.dragging { opacity: 0.4; }
tr.drag-over { outline: 2px dashed #4C72B0; }

.empty-note {
  padding: 12px 16px;
  color: #667;
  font-size: 13px;
}

</style>
</head>

<body>
<div class="container" id="app"></div>

<script>
const datasets = __DATA__;

let currentRows = null;
let currentRender = null;

function makeKey(r) {
  return r.roi_name + "||" + r.clinical_goal;
}

function saveOrder(file) {
  if (!currentRows) return;
  const txt = currentRows.map(makeKey).join("\\n");

  fetch("/save", {
    method: "POST",
    body: JSON.stringify({ file: file, data: txt })
  });
}

function loadOrder(file) {
  if (!currentRows || !currentRender) return;

  fetch("/load?file=" + encodeURIComponent(file))
    .then(r => {
      if (!r.ok) throw 0;
      return r.text();
    })
    .then(text => {
      const order = text.split(/\\r?\\n/).filter(Boolean);

      const map = new Map(currentRows.map(r => [makeKey(r), r]));
      const newRows = [];

      order.forEach(k => {
        if (map.has(k)) {
          newRows.push(map.get(k));
          map.delete(k);
        }
      });

      map.forEach(v => newRows.push(v));

      currentRows.splice(0, currentRows.length, ...newRows);
      currentRender();
    })
    .catch(() => alert("No saved order"));
}

function heat(v) {
  v = Math.max(0, Math.min(1, v));
  let r, g;

  if (v < 0.5) {
    r = 255;
    g = Math.round(255 * (v / 0.5));
  } else {
    g = 255;
    r = Math.round(255 * (1 - (v - 0.5) / 0.5));
  }

  return `rgba(${r}, ${g}, 0, 0.35)`;
}

function compute(rows, weights, probMass) {
  let running = new Array(weights.length).fill(true);

  return rows.map(r => {
    const prg = r.success.reduce((a, x, i) => a + (x ? weights[i] : 0), 0);
    const pr = prg / probMass;

    running = running.map((x, i) => x && r.success[i]);

    const cpr = running.reduce((a, x, i) => a + (x ? weights[i] : 0), 0) / probMass;

    return { ...r, pr, prg, cpr };
  });
}

function renderDataset(ds) {
  const card = document.createElement("div");
  card.className = "card";

  card.innerHTML = `
    <div class="card-header">
      ${ds.key} - ${(ds.probability * 100).toFixed(1)}%
    </div>

    <div class="section-title">
      Probabilistic objectives
      <span class="section-tools">
        <button class="save-order">Save order</button>
        <button class="load-order">Load order</button>
      </span>
    </div>
    <div class="table-wrap" id="prob-wrap">
      <table>
        <thead>
          <tr>
            <th style="width:30px">Drag</th>
            <th>ROI</th>
            <th>Clinical Goal</th>
            <th>Nominal</th>
            <th class="num">Passing</th>
            <th class="num">Global</th>
            <th class="num">Cumulative</th>
          </tr>
        </thead>
        <tbody id="prob-body"></tbody>
      </table>
    </div>

    <div class="section-title">Non-probabilistic objectives</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ROI</th>
            <th>Clinical Goal</th>
            <th>Nominal</th>
          </tr>
        </thead>
        <tbody id="nominal-body"></tbody>
      </table>
    </div>
  `;

  // Bind through the dataset object: a Windows path pasted into an inline
  // onclick string would have its backslashes eaten as escape characters,
  // which made the server write order.txt to the drive root.
  card.querySelector(".save-order").onclick = () => saveOrder(ds.file_path);
  card.querySelector(".load-order").onclick = () => loadOrder(ds.file_path);

  const probTbody = card.querySelector("#prob-body");
  const nominalTbody = card.querySelector("#nominal-body");
  const probWrap = card.querySelector("#prob-wrap");

  let probRows = ds.prob_rows.map(r => ({...r}));
  const nominalRows = ds.nominal_rows.map(r => ({...r}));

  currentRows = probRows;

  let draggedIndex = null;

  function renderProbTable() {
    probTbody.innerHTML = "";

    if (!probRows.length) {
      probWrap.innerHTML = '<div class="empty-note">No probabilistic objectives in this file.</div>';
      return;
    }

    const computed = compute(probRows, ds.weights, ds.probability);

    computed.forEach((r, i) => {
      const tr = document.createElement("tr");
      tr.draggable = true;

      const cls =
        r.nominal_passed === true ? "pass" :
        r.nominal_passed === false ? "fail" : "";

      tr.innerHTML = `
        <td>⋮</td>
        <td>${r.roi_name}</td>
        <td>${r.clinical_goal}</td>
        <td class="${cls}">${r.nominal}</td>
        <td class="num" style="background:${heat(r.pr)}">${r.pr.toFixed(3)}</td>
        <td class="num" style="background:${heat(r.prg)}">${r.prg.toFixed(3)}</td>
        <td class="num" style="background:${heat(r.cpr)}">${r.cpr.toFixed(3)}</td>
      `;

      tr.addEventListener("dragstart", () => {
        draggedIndex = i;
        tr.classList.add("dragging");
      });

      tr.addEventListener("dragend", () => {
        draggedIndex = null;
        tr.classList.remove("dragging");
        document.querySelectorAll(".drag-over")
          .forEach(e => e.classList.remove("drag-over"));
      });

      tr.addEventListener("dragover", e => {
        e.preventDefault();
        tr.classList.add("drag-over");
      });

      tr.addEventListener("dragleave", () => {
        tr.classList.remove("drag-over");
      });

      tr.addEventListener("drop", e => {
        e.preventDefault();
        tr.classList.remove("drag-over");

        if (draggedIndex === null || draggedIndex === i) return;

        const item = probRows.splice(draggedIndex, 1)[0];
        probRows.splice(i, 0, item);

        renderProbTable();
      });

      probTbody.appendChild(tr);
    });
  }

  function renderNominalTable() {
    nominalTbody.innerHTML = "";

    if (!nominalRows.length) {
      nominalTbody.innerHTML = '<tr><td colspan="3" class="empty-note">No non-probabilistic objectives in this file.</td></tr>';
      return;
    }

    nominalRows.forEach(r => {
      const tr = document.createElement("tr");

      const cls =
        r.nominal_passed === true ? "pass" :
        r.nominal_passed === false ? "fail" : "";

      tr.innerHTML = `
        <td>${r.roi_name}</td>
        <td>${r.clinical_goal}</td>
        <td class="${cls}">${r.nominal}</td>
      `;

      nominalTbody.appendChild(tr);
    });
  }

  currentRender = renderProbTable;
  renderProbTable();
  renderNominalTable();

  return card;
}

const app = document.getElementById("app");
datasets.forEach(ds => app.appendChild(renderDataset(ds)));

</script>
</body>
</html>
"""


# ---------------------------
# Server
# ---------------------------
def launch(dataset):
    html = HTML.replace("__DATA__", json.dumps([dataset]))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/load"):
                q = parse_qs(urlparse(self.path).query)
                file = q.get("file", [None])[0]

                if not file:
                    self.send_response(400)
                    self.end_headers()
                    return

                p = Path(file)
                order_file = p.parent / "order.txt"

                if not order_file.exists():
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.end_headers()
                self.wfile.write(order_file.read_text().encode())
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def do_POST(self):
            if self.path == "/save":
                length = int(self.headers["Content-Length"])
                data = json.loads(self.rfile.read(length))

                p = Path(data["file"])
                order_file = p.parent / "order.txt"
                order_file.write_text(data["data"])

                self.send_response(200)
                self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f"http://{server.server_address[0]}:{server.server_address[1]}")

    return server


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    folder = Path("Results")

    results = load_results(folder)
    payloads = build_payload(results)

    servers = []
    for ds in payloads:
        servers.append(launch(ds))
        time.sleep(0.2)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for s in servers:
            s.shutdown()
