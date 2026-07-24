let reviewers = [];
let filtered = [];
let semanticActive = false; // true once a query abstract has been embedded and applied

// Lazily-loaded transformers.js feature extractor (bge-small, q8).
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
const BGE_MODEL = "Xenova/bge-small-en-v1.5";
const BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
let extractorPromise = null; // Promise<extractor>

const decoder = new TextDecoder();
const unlockForm = document.querySelector("#unlockForm");
const password = document.querySelector("#password");
const gate = document.querySelector("#gate");
const gateStatus = document.querySelector("#gateStatus");
const app = document.querySelector("#app");
const meta = document.querySelector("#meta");
const query = document.querySelector("#query");
const hideExcluded = document.querySelector("#hideExcluded");
const rowsEl = document.querySelector("#rows");
const countEl = document.querySelector("#count");
const detailEl = document.querySelector("#detail");
const matchColHeader = document.querySelector("#matchCol");

const abstract = document.querySelector("#abstract");
const findReviewers = document.querySelector("#findReviewers");
const clearSemantic = document.querySelector("#clearSemantic");
const semanticStatus = document.querySelector("#semanticStatus");

function b64ToBytes(value) {
  return Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
}

async function deriveKey(passphrase, salt, iterations) {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(passphrase),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"],
  );
}

async function unlock(passphrase) {
  const response = await fetch("reviewers.enc.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load reviewers.enc.json");
  const payload = await response.json();
  const key = await deriveKey(passphrase, b64ToBytes(payload.salt), payload.iterations);
  const ciphertext = b64ToBytes(payload.ciphertext);
  const tag = b64ToBytes(payload.tag);
  const combined = new Uint8Array(ciphertext.length + tag.length);
  combined.set(ciphertext);
  combined.set(tag, ciphertext.length);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: b64ToBytes(payload.iv), tagLength: 128 },
    key,
    combined,
  );
  return JSON.parse(decoder.decode(plaintext));
}

// ---- Embeddings ------------------------------------------------------------

// Dequantize an int8-quantized reviewer vector into a Float32Array.
// Stored as base64 int8 bytes with a per-vector scale: value = (int8 / 127) * scale.
function decodeEmbedding(row) {
  if (!row.embedding) return null;
  const bytes = b64ToBytes(row.embedding);
  const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scale = Number(row.embedding_scale || 1);
  const vec = new Float32Array(signed.length);
  let norm = 0;
  for (let i = 0; i < signed.length; i += 1) {
    const v = (signed[i] / 127) * scale;
    vec[i] = v;
    norm += v * v;
  }
  return { vec, norm: Math.sqrt(norm) || 1 };
}

// Attach decoded embeddings to rows once, after unlock.
function prepareEmbeddings() {
  for (const row of reviewers) {
    row._emb = decodeEmbedding(row);
  }
}

function cosine(queryVec, queryNorm, emb) {
  if (!emb) return -1;
  const { vec, norm } = emb;
  let dot = 0;
  for (let i = 0; i < vec.length; i += 1) dot += vec[i] * queryVec[i];
  return dot / (queryNorm * norm);
}

async function getExtractor(onProgress) {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import(TRANSFORMERS_URL);
      env.allowLocalModels = false; // always fetch from the Hub CDN
      return pipeline("feature-extraction", BGE_MODEL, {
        dtype: "q8",
        progress_callback: onProgress,
      });
    })();
  }
  return extractorPromise;
}

async function embedQuery(text, onProgress) {
  const extractor = await getExtractor(onProgress);
  // CLS pooling + normalize is required for bge parity (mean pooling degrades quality).
  const output = await extractor(BGE_QUERY_PREFIX + text, { pooling: "cls", normalize: true });
  const data = Float32Array.from(output.data);
  let norm = 0;
  for (let i = 0; i < data.length; i += 1) norm += data[i] * data[i];
  return { vec: data, norm: Math.sqrt(norm) || 1 };
}

async function runSemanticSearch() {
  const text = abstract.value.trim();
  if (text.length < 20) {
    semanticStatus.textContent = "Paste a longer abstract (at least a sentence or two).";
    return;
  }
  findReviewers.disabled = true;
  const t0 = performance.now();
  let downloadNoted = false;
  try {
    const q = await embedQuery(text, (p) => {
      // p: { status, file, progress, loaded, total }
      if (p && p.status === "progress" && p.file && /\.onnx/.test(p.file)) {
        downloadNoted = true;
        const pct = p.progress ? p.progress.toFixed(0) : "0";
        semanticStatus.textContent = `Downloading embedding model (one time, ~34MB): ${pct}%`;
      } else if (p && p.status === "progress") {
        downloadNoted = true;
        semanticStatus.textContent = "Downloading embedding model (one time, ~34MB)...";
      } else if (p && p.status === "ready" && !downloadNoted) {
        semanticStatus.textContent = "Embedding your abstract...";
      }
    });
    const tModel = performance.now();
    let scored = 0;
    for (const row of reviewers) {
      if (row._emb) {
        row._score = cosine(q.vec, q.norm, row._emb);
        scored += 1;
      } else {
        row._score = -1;
      }
    }
    const tQuery = performance.now();
    semanticActive = true;
    clearSemantic.hidden = false;
    matchColHeader.hidden = false;
    const modelMs = Math.round(tModel - t0);
    const scoreMs = Math.round(tQuery - tModel);
    semanticStatus.textContent =
      `Ranked ${scored} contributors by match. ` +
      `Model+embed ${modelMs} ms, scoring ${scoreMs} ms.`;
    applyFilters();
  } catch (error) {
    console.error(error);
    semanticStatus.textContent = "Semantic search failed to load the model. See console for details.";
  } finally {
    findReviewers.disabled = false;
  }
}

function clearSemanticSearch() {
  semanticActive = false;
  clearSemantic.hidden = true;
  matchColHeader.hidden = true;
  semanticStatus.textContent = "";
  for (const row of reviewers) row._score = undefined;
  applyFilters();
}

// ---- Filtering + rendering -------------------------------------------------

function searchable(row) {
  return [
    row.name,
    row.email,
    row.affiliation,
    row.orcid_employments,
    row.forrt_projects,
    row.forrt_roles,
    row.research_topics,
    row.methods_tags,
    row.disciplines,
    row.llm_profile,
    row.fit_notes,
  ].join(" ").toLowerCase();
}

function applyFilters() {
  const terms = query.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  filtered = reviewers.filter((row) => {
    if (hideExcluded.checked && row.exclude_reason) return false;
    const haystack = searchable(row);
    return terms.every((term) => haystack.includes(term));
  });
  if (semanticActive) {
    filtered = filtered.slice().sort((a, b) => (b._score ?? -1) - (a._score ?? -1));
  }
  renderRows();
}

function short(text, length = 220) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length > length ? `${clean.slice(0, length - 1)}...` : clean;
}

function pills(value, limit = 4) {
  return String(value || "")
    .split(";")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, limit)
    .map((t) => `<span class="tag">${escapeHtml(t)}</span>`)
    .join("");
}

function evidenceBadge(row) {
  const s = String(row.evidence_strength || "").trim().toLowerCase();
  if (!s) return "";
  return `<span class="ev ev-${escapeHtml(s)}">${escapeHtml(s)} evidence</span>`;
}

let selectedOrcid = null;

function renderRows() {
  countEl.textContent = semanticActive
    ? `${filtered.length} of ${reviewers.length} contributors · ranked by match`
    : `${filtered.length} of ${reviewers.length} contributors`;
  rowsEl.innerHTML = "";
  for (const row of filtered.slice(0, 300)) {
    const tr = document.createElement("tr");
    if (row.orcid && row.orcid === selectedOrcid) tr.classList.add("selected");
    let matchCell = "";
    if (semanticActive) {
      const score = typeof row._score === "number" ? row._score : -1;
      const pct = score >= 0 ? Math.round(score * 100) : 0;
      const thin = row.doc_fallback === "1"
        ? `<span class="thin" title="Match rests on thin fallback text, not published abstracts.">thin profile</span>`
        : "";
      matchCell = `<td class="c-match"><div class="meter"><span style="width:${pct}%"></span></div><div class="meter-val">${pct}%</div>${thin}</td>`;
    }
    const tagsLine = pills([row.disciplines, row.methods_tags].filter(Boolean).join("; "), 4);
    tr.innerHTML = `
      ${matchCell}
      <td><div class="name">${escapeHtml(row.name)}</div><div class="orcid">${escapeHtml(row.orcid || "")}</div>${evidenceBadge(row)}</td>
      <td class="c-affil muted">${escapeHtml(short(row.affiliation || row.orcid_employments || "—", 110))}</td>
      <td><div class="snippet">${escapeHtml(short(row.llm_profile || row.fit_notes || row.forrt_projects, 190))}</div>${tagsLine ? `<div class="tagline">${tagsLine}</div>` : ""}</td>
    `;
    tr.addEventListener("click", () => selectRow(row));
    rowsEl.appendChild(tr);
  }
}

function selectRow(row) {
  selectedOrcid = row.orcid || null;
  for (const tr of rowsEl.children) tr.classList.remove("selected");
  renderRows();
  renderDetail(row);
  if (window.matchMedia("(max-width: 960px)").matches) {
    detailEl.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    detailEl.scrollTop = 0;
  }
}

function chipList(value, limit) {
  const inner = pills(value, limit);
  return inner ? `<div class="chips">${inner}</div>` : "";
}

function renderDetail(row) {
  const topics = chipList(row.research_topics, 18);
  const methods = chipList(row.methods_tags, 12);
  const disciplines = chipList(row.disciplines, 6);
  const matchReadout = semanticActive && typeof row._score === "number"
    ? `<div class="match-readout">
         <span class="pct">${Math.round(row._score * 100)}%</span>
         <div>
           <div class="meter meter-lg"><span style="width:${Math.round(row._score * 100)}%"></span></div>
           <span class="lbl">match to your abstract${row.doc_fallback === "1" ? " · thin profile, matched on fallback text" : ""}</span>
         </div>
       </div>`
    : "";
  const email = String(row.email || "").trim();
  const contact = email
    ? `<p class="contact"><a href="mailto:${escapeHtml(email)}">${escapeHtml(email)}</a></p>`
    : "";
  detailEl.innerHTML = `
    <h2>${escapeHtml(row.name)}</h2>
    <p class="sub">${escapeHtml(row.affiliation || row.orcid_employments || "No affiliation recorded")}</p>
    ${contact}
    ${evidenceBadge(row)}
    ${matchReadout}
    ${row.exclude_reason ? `<p class="excluded"><strong>Excluded:</strong> ${escapeHtml(row.exclude_reason)}</p>` : ""}
    <h3>Profile</h3>
    <p>${escapeHtml(row.llm_profile || "No profile generated yet.")}</p>
    ${disciplines ? `<h3>Disciplines</h3>${disciplines}` : ""}
    ${topics ? `<h3>Research topics</h3>${topics}` : ""}
    ${methods ? `<h3>Methods</h3>${methods}` : ""}
    ${row.forrt_projects ? `<h3>Projects &amp; roles</h3><p>${escapeHtml(row.forrt_projects)}</p>${row.forrt_roles ? `<p class="muted">${escapeHtml(row.forrt_roles)}</p>` : ""}` : ""}
    ${row.fit_notes ? `<h3>Reviewer fit</h3><p>${escapeHtml(row.fit_notes)}</p>` : ""}
    <h3>Evidence</h3>
    <p class="evidence-note"><strong>${escapeHtml(row.central_works_count || "0")}</strong> authorship-qualified abstracts used for profiling · <strong>${escapeHtml(row.works_orcid_confirmed || "0")}</strong> ORCID-confirmed works of <strong>${escapeHtml(row.works_total_attributed || "0")}</strong> attributed on OpenAlex.</p>
    ${row.orcid ? `<h3>Links</h3><a class="orcid-link" href="https://orcid.org/${encodeURIComponent(row.orcid)}" target="_blank" rel="noreferrer">ORCID ${escapeHtml(row.orcid)}</a>` : ""}
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatGeneratedAt(value) {
  if (!value) return "unknown date";
  const d = new Date(value);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

const PW_KEY = "forrt_reviewer_pw";
const headerStatus = document.querySelector("#headerStatus");
const lockButton = document.querySelector("#lock");

function storePassword(value) {
  try { localStorage.setItem(PW_KEY, value); } catch {}
}
function readStoredPassword() {
  try { return localStorage.getItem(PW_KEY); } catch { return null; }
}
function clearStoredPassword() {
  try { localStorage.removeItem(PW_KEY); } catch {}
}

async function attemptUnlock(passphrase, { fromStorage = false } = {}) {
  gateStatus.textContent = "Decrypting…";
  try {
    const payload = await unlock(passphrase);
    storePassword(passphrase);
    if (window.PasswordCredential) {
      try {
        await navigator.credentials.store(new PasswordCredential({ id: "forrt-team", password: passphrase }));
      } catch {}
    }
    reviewers = payload.rows || [];
    filtered = reviewers;
    prepareEmbeddings();
    const withEmb = reviewers.filter((r) => r._emb).length;
    meta.textContent = `${reviewers.length} contributors · ${withEmb} embedded · updated ${formatGeneratedAt(payload.generatedAt)}`;
    headerStatus.hidden = false;
    gate.hidden = true;
    app.hidden = false;
    gateStatus.textContent = "";
    applyFilters();
  } catch (error) {
    if (fromStorage) {
      clearStoredPassword(); // stale/rotated password — fall back to the gate
      gateStatus.textContent = "";
    } else {
      gateStatus.textContent = "That password didn't unlock the data. Check it and try again.";
    }
  }
}

unlockForm.addEventListener("submit", (event) => {
  event.preventDefault();
  attemptUnlock(password.value);
});

lockButton.addEventListener("click", () => {
  clearStoredPassword();
  location.reload();
});

// Auto-unlock if a password was saved on this device.
const saved = readStoredPassword();
if (saved) {
  password.value = saved;
  attemptUnlock(saved, { fromStorage: true });
}

[query, hideExcluded].forEach((el) => {
  el.addEventListener("input", applyFilters);
});

findReviewers.addEventListener("click", runSemanticSearch);
clearSemantic.addEventListener("click", clearSemanticSearch);
