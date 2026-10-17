"use strict";

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderConversationalMarkdown(text) {
  if (!text) return "";
  const escaped = escapeHtml(text);
  // Render fenced code blocks first with a 1-click Copy button
  const parts = escaped.split(/```([\s\S]*?)```/g);
  const renderedParts = parts.map((chunk, idx) => {
    if (idx % 2 === 1) {
      const codeBody = chunk.trim();
      return `
        <div class="code-block-wrap">
          <div class="code-block-bar">
            <span class="code-block-lang">CODE</span>
            <button type="button" class="copy-code-btn" data-code="${codeBody}">Copy</button>
          </div>
          <div class="code-snippet">${codeBody}</div>
        </div>
      `;
    }
    // Inline formatting: **bold**, *italic*, `code`
    let html = chunk
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");

    // Convert bullet lists and paragraphs
    const lines = html.split("\n");
    let out = "";
    let inList = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("- ")) {
        if (!inList) {
          out += "<ul>";
          inList = true;
        }
        out += `<li>${trimmed.slice(2)}</li>`;
      } else {
        if (inList) {
          out += "</ul>";
          inList = false;
        }
        if (trimmed.length > 0) {
          out += `<p>${trimmed}</p>`;
        }
      }
    }
    if (inList) out += "</ul>";
    return out;
  });
  return renderedParts.join("");
}

async function apiRequest(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json();
  if (!payload.ok) {
    throw new Error(payload.error || `Request failed with HTTP ${res.status}`);
  }
  return payload.data;
}

let cachedStatus = null;

// ---------------------------------------------------------------------------
// Status & Sidebar Refresh
// ---------------------------------------------------------------------------

async function refreshSystemStatus() {
  try {
    const data = await apiRequest("/api/status");
    cachedStatus = data;

    const conn = data.model_connectivity || {};
    const dot = document.getElementById("connectivity-dot");
    const label = document.getElementById("connectivity-label");

    dot.className = "dot";
    if (conn.status === "CONNECTED") {
      dot.classList.add("ok");
    } else if (
      conn.status === "MISSING_CREDENTIALS" ||
      conn.status === "LOCAL_ONLY" ||
      conn.status === "DETERMINISTIC_TEST_DOUBLE"
    ) {
      dot.classList.add("warn");
    } else {
      dot.classList.add("err");
    }
    label.textContent = `${conn.provider || "model"}:${conn.model_id || ""} [${conn.status || "UNKNOWN"}]`;

    document.getElementById("caps-count-label").textContent =
      `${data.capabilities_available}/${data.capabilities_registered} Capabilities Operational`;

    document.getElementById("memory-count-label").textContent =
      `${data.persisted_memories_count} Memories | ${data.persisted_experiences_count} Experiences`;

    renderRecentTasksSidebar(data.recent_tasks || []);
  } catch (err) {
    document.getElementById("connectivity-label").textContent = `Status Error: ${err.message}`;
  }
}

function renderRecentTasksSidebar(tasks) {
  const container = document.getElementById("recent-tasks-list");
  if (!tasks || tasks.length === 0) {
    container.innerHTML = `<div class="empty-hint">No persisted tasks yet.</div>`;
    return;
  }
  container.innerHTML = tasks
    .map(
      (t) => `
      <div class="mini-task-item" data-task-id="${escapeHtml(t.task_id)}">
        <div class="mini-task-top">
          <span>${escapeHtml(t.task_id)}</span>
          <span class="badge badge-${escapeHtml(t.outcome)}">${escapeHtml(t.outcome)}</span>
        </div>
        <div class="mini-task-instr">${escapeHtml(t.original_instruction)}</div>
      </div>
    `
    )
    .join("");

  container.querySelectorAll(".mini-task-item").forEach((el) => {
    el.addEventListener("click", () => {
      const tid = el.getAttribute("data-task-id");
      switchTab("tasks");
      inspectTaskById(tid);
    });
  });
}

// ---------------------------------------------------------------------------
// Task Card Renderer (used in both Chat Stream and Task Inspector)
// ---------------------------------------------------------------------------

function renderTaskCard(task, showResumeButton = true, expandTraceByDefault = false) {
  const actions = task.planned_actions || [];
  const verifications = task.verification_results || [];
  const liveTests = task.live_run_results || [];
  const errors = task.errors || [];
  const recoveries = task.recovery_attempts || [];
  const decisions = task.decisions || [];
  const outputs = task.relevant_outputs || {};

  const canResume =
    showResumeButton &&
    (task.current_execution_state === "INTERRUPTED" ||
      task.final_outcome === "PARTIAL_SUCCESS" ||
      task.final_outcome === "FAILED");

  const stepsHtml = actions
    .map((act) => {
      const outVal = outputs[act.step_id];
      const outSnippet =
        outVal !== undefined
          ? `<div class="code-snippet">${escapeHtml(JSON.stringify(outVal, null, 2))}</div>`
          : "";
      const errSnippet = act.error
        ? `<div class="code-snippet" style="border-color: var(--danger); color: #ffa198;">Error: ${escapeHtml(act.error)}</div>`
        : "";
      return `
        <div class="step-row">
          <div class="card-item-header">
            <span class="mono"><strong>${escapeHtml(act.step_id)}</strong> &bull; <code>${escapeHtml(act.capability_id)}</code> &mdash; ${escapeHtml(act.description)}</span>
            <span class="badge badge-${escapeHtml(act.status)}">${escapeHtml(act.status)}</span>
          </div>
          ${outSnippet}
          ${errSnippet}
        </div>
      `;
    })
    .join("");

  const verifHtml = verifications
    .map(
      (v) => `
      <div class="eval-row">
        <div class="card-item-header">
          <span><strong>${escapeHtml(v.requirement)}</strong> <span class="mono" style="color:var(--text-secondary)">(${escapeHtml(v.method)})</span></span>
          <span class="badge badge-${escapeHtml(v.status)}">${escapeHtml(v.status)}</span>
        </div>
        <div style="color: var(--text-secondary); margin-top: 3px;">Observed: ${escapeHtml(v.observed_result)}</div>
        ${
          v.failure_explanation
            ? `<div style="color: #ffa198; margin-top: 3px;">Explanation: ${escapeHtml(v.failure_explanation)}</div>`
            : ""
        }
      </div>
    `
    )
    .join("");

  const liveHtml = liveTests
    .map(
      (l) => `
      <div class="eval-row">
        <div class="card-item-header">
          <span class="mono"><strong>${escapeHtml(l.target_component)}</strong> &bull; <code>${escapeHtml(l.executed_operation)}</code></span>
          <span class="badge badge-${escapeHtml(l.status)}">${escapeHtml(l.status)}</span>
        </div>
        <div class="mono" style="color: var(--text-secondary); font-size: 12px; margin-top: 3px;">
          exit_code=${escapeHtml(String(l.exit_code))} | expected_behavior=${escapeHtml(String(l.expected_behavior_occurred))} | ${Number(l.duration_ms || 0).toFixed(1)}ms
        </div>
        ${l.stdout ? `<div class="code-snippet">${escapeHtml(l.stdout)}</div>` : ""}
        ${l.stderr ? `<div class="code-snippet" style="color:#ffa198">${escapeHtml(l.stderr)}</div>` : ""}
        ${l.limitations ? `<div style="color:var(--warning); margin-top:4px;">Limitation: ${escapeHtml(l.limitations)}</div>` : ""}
      </div>
    `
    )
    .join("");

  const errorsHtml =
    errors.length > 0
      ? `
      <div class="task-section">
        <h4>Errors &amp; Targeted Recovery (${errors.length} Error(s), ${recoveries.length} Recovery Attempt(s))</h4>
        ${errors
          .map(
            (e) => `
            <div class="eval-row">
              <div class="card-item-header">
                <span class="mono"><strong>${escapeHtml(e.error_type)}</strong> (step=${escapeHtml(e.step_id || "n/a")})</span>
                <span class="badge badge-${e.recovered ? "PASSED" : "FAILED"}">${e.recovered ? "RECOVERED" : "UNRECOVERED"}</span>
              </div>
              <div>${escapeHtml(e.message)}</div>
            </div>
          `
          )
          .join("")}
        ${recoveries
          .map(
            (r) => `
            <div class="eval-row">
              <div class="card-item-header">
                <span class="mono">Recovery Attempt on <strong>${escapeHtml(r.step_id)}</strong></span>
                <span class="badge badge-${r.succeeded ? "PASSED" : "FAILED"}">${r.succeeded ? "SUCCEEDED" : "FAILED"}</span>
              </div>
              <div>Cause: ${escapeHtml(r.probable_cause)}</div>
              <div>Correction: <code>${escapeHtml(r.selected_correction)}</code></div>
            </div>
          `
          )
          .join("")}
      </div>
    `
      : "";

  const decisionsHtml =
    decisions.length > 0
      ? `
      <div class="task-section">
        <h4>Decisions &amp; Experience Applied</h4>
        <ul style="padding-left: 18px; color: var(--text-secondary);">
          ${decisions.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
        </ul>
      </div>
    `
      : "";

  const distilledSpec = task.distilled_spec || null;
  const distilledHtml =
    distilledSpec && distilledSpec.is_build_intent
      ? `
      <div class="task-section" style="border-left: 3px solid var(--accent); padding-left: 10px;">
        <h4>Distilled Concrete Build Specification ${
          distilledSpec.is_messy_input
            ? `<span class="badge badge-PASSED" style="margin-left:6px;">DISTILLED FROM MESSY WRITING</span>`
            : ""
        }</h4>
        <div style="margin-bottom: 4px;"><strong>What User Wants to Build:</strong> ${escapeHtml(distilledSpec.what_user_wants_to_build)}</div>
        <div style="margin-bottom: 4px;"><strong>Concrete Instruction:</strong> ${escapeHtml(distilledSpec.concrete_instruction)}</div>
        ${
          (distilledSpec.extracted_functions || []).length > 0
            ? `<div style="margin-bottom: 4px;"><strong>Identified Functions:</strong> ${(distilledSpec.extracted_functions || [])
                .map((fn) => `<code>${escapeHtml(fn)}()</code>`)
                .join(", ")}</div>`
            : ""
        }
        ${
          (distilledSpec.ambiguities_resolved || []).length > 0
            ? `<div style="color: var(--text-secondary); font-size: 12px;"><strong>Ambiguities Resolved:</strong><ul style="padding-left:18px; margin-top:3px;">${(
                distilledSpec.ambiguities_resolved || []
              )
                .map((a) => `<li>${escapeHtml(a)}</li>`)
                .join("")}</ul></div>`
            : ""
        }
      </div>
    `
      : "";

  const isConversational = task.interaction_mode === "conversational";
  const replyHtml = task.assistant_response
    ? `<div class="conversational-reply">${renderConversationalMarkdown(task.assistant_response)}</div>`
    : `<div class="conversational-reply"><p>${escapeHtml(task.outcome_summary)}</p></div>`;

  // Compact tool & proof pills (Hermes / OpenClaw style, unique Agent 911 aesthetic)
  const nonChatActions = actions.filter((a) => a.capability_id !== "chat.respond");
  const toolChipsHtml = nonChatActions
    .slice(0, 4)
    .map(
      (a) =>
        `<span class="tool-chip" title="${escapeHtml(a.description)}">⚡ ${escapeHtml(a.capability_id)}</span>`
    )
    .join("");

  const verifPassed = verifications.filter((v) => v.status === "PASSED").length;
  const livePassed = liveTests.filter((l) => l.status === "PASSED").length;
  const proofChipsHtml = !isConversational
    ? `
      ${
        verifications.length > 0
          ? `<span class="proof-chip proof-chip-ok">✓ Verified (${verifPassed}/${verifications.length})</span>`
          : ""
      }
      ${
        liveTests.length > 0
          ? `<span class="proof-chip proof-chip-ok">✓ Live-Tested (${livePassed}/${liveTests.length})</span>`
          : ""
      }
    `
    : `<span class="proof-chip">💬 Chat Reply</span>`;

  const distilledBannerChip =
    distilledSpec && distilledSpec.is_build_intent && distilledSpec.is_messy_input
      ? `
      <div class="distilled-compact-banner">
        <span class="distilled-badge">🎯 BUILD SPEC</span>
        <span class="distilled-text">${escapeHtml(distilledSpec.what_user_wants_to_build)}</span>
      </div>
    `
      : "";

  return `
    <div data-card-task-id="${escapeHtml(task.task_id)}">
      ${distilledBannerChip}
      ${replyHtml}

      <details class="agent-trace-drawer" ${expandTraceByDefault ? "open" : ""}>
        <summary>
          <div class="trace-pill-row">
            ${toolChipsHtml}
            ${proofChipsHtml}
            <span class="trace-toggle-hint">Inspect Trace ▸</span>
          </div>
          <span class="badge badge-${escapeHtml(task.final_outcome)}">${escapeHtml(task.final_outcome)}</span>
        </summary>

        <div class="task-card">
          <div class="task-card-summary">
            <div>
              <span class="mono"><strong>${escapeHtml(task.task_id)}</strong></span>
              <span style="color:var(--text-secondary); margin-left:8px;">Stage: <code>${escapeHtml(task.current_execution_state)}</code></span>
              <span style="color:var(--text-secondary); margin-left:8px;">Strategy: <code>${escapeHtml(task.selected_strategy || "DIRECT_EXECUTION")}</code></span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              ${
                canResume
                  ? `<button class="btn btn-secondary btn-sm resume-task-btn" data-resume-id="${escapeHtml(task.task_id)}" type="button">Resume Task</button>`
                  : ""
              }
              <span class="badge badge-${escapeHtml(task.final_outcome)}">Outcome: ${escapeHtml(task.final_outcome)}</span>
            </div>
          </div>

          <div style="color: var(--text-primary); font-weight: 500;">
            ${escapeHtml(task.outcome_summary)}
          </div>

          ${distilledHtml}

          ${decisionsHtml}

          <div class="task-section">
            <h4>Planned &amp; Executed Actions (${actions.length})</h4>
            ${stepsHtml || `<div class="empty-hint">No actions planned.</div>`}
          </div>

          ${errorsHtml}

          <div class="task-section">
            <h4>Formal Evaluation Mechanism 1: Verification (${verifications.length})</h4>
            ${verifHtml || `<div class="empty-hint">No verification checks recorded.</div>`}
          </div>

          <div class="task-section">
            <h4>Formal Evaluation Mechanism 2: Live Run Testing (${liveTests.length})</h4>
            ${liveHtml || `<div class="empty-hint">No live run tests recorded.</div>`}
          </div>

          <div class="mono" style="font-size: 11px; color: var(--text-secondary);">
            Associated Experience ID(s): ${escapeHtml((task.associated_experience_ids || []).join(", ") || "none")}
            • Resource Metrics: ${escapeHtml(JSON.stringify(task.resource_metrics || {}))}
          </div>
        </div>
      </details>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Chat Console Submission
// ---------------------------------------------------------------------------

function appendChatMessage(role, headerText, metaText, bodyHtml) {
  const stream = document.getElementById("chat-stream");
  const msg = document.createElement("div");
  msg.className = `chat-msg ${role === "user" ? "user-msg" : "assistant-msg"}`;
  const authorMarkup =
    role === "user"
      ? `<span class="msg-author">${escapeHtml(headerText)}</span>`
      : `<div class="agent-avatar-chip"><span class="avatar-dot"></span><span class="msg-author">${escapeHtml(headerText)}</span></div>`;
  msg.innerHTML = `
    <div class="msg-header">
      ${authorMarkup}
      <span class="msg-meta">${escapeHtml(metaText)}</span>
    </div>
    <div class="msg-body">${bodyHtml}</div>
  `;
  stream.appendChild(msg);
  stream.scrollTop = stream.scrollHeight;
  bindResumeButtons(msg);
  bindCopyCodeButtons(msg);
  return msg;
}

function bindCopyCodeButtons(rootEl) {
  rootEl.querySelectorAll(".copy-code-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const raw = btn.getAttribute("data-code") || "";
      try {
        await navigator.clipboard.writeText(raw);
        btn.textContent = "Copied ✓";
        setTimeout(() => {
          btn.textContent = "Copy";
        }, 1500);
      } catch (_e) {
        btn.textContent = "Copied";
      }
    });
  });
}

function bindResumeButtons(rootEl) {
  rootEl.querySelectorAll(".resume-task-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const tid = btn.getAttribute("data-resume-id");
      btn.disabled = true;
      btn.textContent = "Resuming...";
      try {
        const resumed = await apiRequest(`/api/tasks/${encodeURIComponent(tid)}/resume`, {
          method: "POST",
        });
        appendChatMessage(
          "assistant",
          "Agent 911",
          `${resumed.task_id} • ${resumed.final_outcome}`,
          renderTaskCard(resumed, true, false)
        );
        await refreshSystemStatus();
      } catch (err) {
        alert(`Failed to resume task: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.textContent = "Resume Task";
      }
    });
  });
}

async function submitChatInstruction(instructionText) {
  const instruction = instructionText.trim();
  if (!instruction) return;

  const inputEl = document.getElementById("chat-input");
  const submitBtn = document.getElementById("chat-submit-btn");
  const statusEl = document.getElementById("composer-status");
  const popover = document.getElementById("distill-preview-popover");
  if (popover) popover.classList.add("hidden");

  appendChatMessage(
    "user",
    "You",
    new Date().toLocaleTimeString(),
    `<p>${escapeHtml(instruction)}</p>`
  );

  inputEl.value = "";
  submitBtn.disabled = true;
  statusEl.textContent = "Agent 911 is thinking & executing...";

  try {
    const task = await apiRequest("/api/chat", {
      method: "POST",
      body: JSON.stringify({ instruction }),
    });
    appendChatMessage(
      "assistant",
      "Agent 911",
      `${task.interaction_mode === "conversational" ? "Chat" : "Build & Execute"} • ${task.task_id}`,
      renderTaskCard(task, true, false)
    );
    await refreshSystemStatus();
  } catch (err) {
    appendChatMessage(
      "assistant",
      "Agent 911 Error",
      "Execution Error",
      `<div class="code-snippet" style="border-color:var(--danger); color:#ffa198;">${escapeHtml(err.message)}</div>`
    );
  } finally {
    submitBtn.disabled = false;
    statusEl.textContent = "Ready • Enter to send";
  }
}

// ---------------------------------------------------------------------------
// Tab 2: Execution History & Task Inspector
// ---------------------------------------------------------------------------

async function loadTasksTab() {
  const container = document.getElementById("tasks-table-container");
  try {
    const data = await apiRequest("/api/tasks?limit=40");
    const tasks = data.tasks || [];
    if (tasks.length === 0) {
      container.innerHTML = `<div class="empty-hint">No tasks recorded in SQLite yet.</div>`;
      return;
    }
    container.innerHTML = tasks
      .map(
        (t) => `
        <div class="card-item task-list-row" style="cursor:pointer;" data-task-id="${escapeHtml(t.task_id)}">
          <div class="card-item-header">
            <span class="mono"><strong>${escapeHtml(t.task_id)}</strong></span>
            <span class="badge badge-${escapeHtml(t.outcome)}">${escapeHtml(t.outcome)}</span>
          </div>
          <div style="font-size:13px;">${escapeHtml(t.original_instruction)}</div>
          <div class="mono" style="font-size:11px; color:var(--text-secondary);">
            Stage: ${escapeHtml(t.stage)} | Updated: ${escapeHtml(t.updated_at)}
          </div>
        </div>
      `
      )
      .join("");

    container.querySelectorAll(".task-list-row").forEach((row) => {
      row.addEventListener("click", () => {
        inspectTaskById(row.getAttribute("data-task-id"));
      });
    });
  } catch (err) {
    container.innerHTML = `<div class="empty-hint">Error loading tasks: ${escapeHtml(err.message)}</div>`;
  }
}

async function inspectTaskById(taskId) {
  const detail = document.getElementById("task-inspector-detail");
  detail.innerHTML = `<div class="empty-hint">Loading task ${escapeHtml(taskId)}...</div>`;
  try {
    const task = await apiRequest(`/api/tasks/${encodeURIComponent(taskId)}`);
    const logs = task.execution_logs || [];
    const logsHtml =
      logs.length > 0
        ? `
        <div class="task-section mt-16">
          <h4>Observability Execution Logs (${logs.length})</h4>
          ${logs
            .map(
              (lg) => `
              <div class="eval-row mono" style="font-size:12px;">
                [${escapeHtml(lg.timestamp)}] step=${escapeHtml(lg.step_id || "-")} cap=${escapeHtml(lg.capability_id || "-")} op=${escapeHtml(lg.operation)} success=${escapeHtml(String(lg.success))} (${Number(lg.duration_ms || 0).toFixed(1)}ms)
              </div>
            `
            )
            .join("")}
        </div>
      `
        : "";
    detail.innerHTML = renderTaskCard(task, true, true) + logsHtml;
    bindResumeButtons(detail);
    bindCopyCodeButtons(detail);
  } catch (err) {
    detail.innerHTML = `<div class="empty-hint">Failed to load task: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Tab 3: Capabilities & Organic Growth
// ---------------------------------------------------------------------------

async function loadCapabilitiesTab() {
  const container = document.getElementById("capabilities-list-container");
  const onlyAvail = document.getElementById("filter-available-caps").checked;
  try {
    const data = await apiRequest(`/api/capabilities?only_available=${onlyAvail}`);
    const caps = data.capabilities || [];
    container.innerHTML = caps
      .map(
        (c) => `
        <div class="card-item">
          <div class="card-item-header">
            <span class="mono"><strong>${escapeHtml(c.identifier)}</strong> ${
          c.is_custom_grown ? `<span class="badge badge-VERIFIED">CUSTOM GROWN</span>` : ""
        }</span>
            <span class="badge badge-${escapeHtml(c.availability)}">${escapeHtml(c.availability)}</span>
          </div>
          <div style="font-weight:600; font-size:13px;">${escapeHtml(c.name)}</div>
          <div style="color:var(--text-secondary); font-size:12px;">${escapeHtml(c.purpose)}</div>
          <div class="mono" style="font-size:11px; color:var(--text-secondary);">
            Required params: ${escapeHtml(((c.input_schema || {}).required || []).join(", ") || "none")}
          </div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    container.innerHTML = `<div class="empty-hint">Failed to load capabilities: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Tab 4: Persistent Memory & Experience
// ---------------------------------------------------------------------------

async function loadMemoriesAndExperiences() {
  const memQuery = document.getElementById("memory-search-query").value.trim();
  const memCat = document.getElementById("memory-search-category").value;
  const expQuery = document.getElementById("experience-search-query").value.trim();

  const memContainer = document.getElementById("memories-list-container");
  const expContainer = document.getElementById("experiences-list-container");

  try {
    const params = new URLSearchParams();
    if (memQuery) params.set("query", memQuery);
    if (memCat) params.set("category", memCat);
    const memData = await apiRequest(`/api/memory?${params.toString()}`);
    const records = memData.records || [];
    memContainer.innerHTML =
      records.length === 0
        ? `<div class="empty-hint">No matching memory records in SQLite.</div>`
        : records
            .map(
              (r) => `
              <div class="card-item">
                <div class="card-item-header">
                  <span class="mono"><strong>${escapeHtml(r.key)}</strong> <span style="color:var(--text-secondary)">[${escapeHtml(r.category)}]</span></span>
                  <span class="badge badge-${escapeHtml(r.confidence)}">${escapeHtml(r.confidence)}</span>
                </div>
                <div>${escapeHtml(r.content)}</div>
                <div class="mono" style="font-size:11px; color:var(--text-secondary);">
                  id=${escapeHtml(r.memory_id)} | source=${escapeHtml(r.source)} | updated=${escapeHtml(r.updated_at)}
                </div>
              </div>
            `
            )
            .join("");
  } catch (err) {
    memContainer.innerHTML = `<div class="empty-hint">Error loading memories: ${escapeHtml(err.message)}</div>`;
  }

  try {
    const expParams = new URLSearchParams();
    if (expQuery) expParams.set("query", expQuery);
    const expData = await apiRequest(`/api/experiences?${expParams.toString()}`);
    const exps = expData.experiences || [];
    expContainer.innerHTML =
      exps.length === 0
        ? `<div class="empty-hint">No matching experience records in SQLite.</div>`
        : exps
            .map(
              (e) => `
              <div class="card-item">
                <div class="card-item-header">
                  <span class="mono"><strong>${escapeHtml(e.experience_id)}</strong> (task=${escapeHtml(e.task_id)})</span>
                  <span class="badge badge-${escapeHtml(e.outcome)}">${escapeHtml(e.outcome)}</span>
                </div>
                <div><strong>Objective:</strong> ${escapeHtml(e.objective)}</div>
                <div class="mono" style="font-size:12px; color:var(--text-secondary);">
                  Capabilities: ${escapeHtml((e.capabilities_used || []).join(", ") || "none")}
                </div>
                ${
                  (e.unexpected_failures || []).length > 0
                    ? `<div style="color:#ffa198; font-size:12px;">Failures: ${escapeHtml(e.unexpected_failures.join("; "))}</div>`
                    : ""
                }
                ${
                  (e.successful_remedies || []).length > 0
                    ? `<div style="color:#3fb950; font-size:12px;">Learned Remedies: ${escapeHtml(e.successful_remedies.join("; "))}</div>`
                    : ""
                }
                ${
                  (e.reusable_procedures || []).length > 0
                    ? `<div class="mono" style="font-size:11px; color:var(--text-secondary);">Procedure: ${escapeHtml(e.reusable_procedures.join(" | "))}</div>`
                    : ""
                }
              </div>
            `
            )
            .join("");
  } catch (err) {
    expContainer.innerHTML = `<div class="empty-hint">Error loading experiences: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Tab Navigation & Event Wiring
// ---------------------------------------------------------------------------

function switchTab(tabName) {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });

  if (tabName === "tasks") loadTasksTab();
  if (tabName === "capabilities") loadCapabilitiesTab();
  if (tabName === "memory") loadMemoriesAndExperiences();
  if (tabName === "models") loadModelsAndProvidersTab();
  if (tabName === "improvements") loadImprovementsTab();
}

async function loadImprovementsTab() {
  const propList = document.getElementById("proposals-list");
  const runList = document.getElementById("self-improve-runs-list");
  if (!propList || !runList) return;

  try {
    const data = await apiRequest("/api/proposals");
    const proposals = data.proposals || [];
    const runs = data.self_improvement_runs || [];

    propList.innerHTML =
      proposals.length === 0
        ? `<div class="empty-hint">No ImprovementProposals recorded yet. Click "Scan History &amp; Propose Improvements" above.</div>`
        : proposals
            .map(
              (p) => `
              <div class="list-card">
                <div class="list-card-header">
                  <span class="mono"><strong>${escapeHtml(p.proposal_id)}</strong> — ${escapeHtml(p.title)}</span>
                  <div style="display:flex; gap:4px;">
                    <span class="badge badge-AVAILABLE">${escapeHtml(p.priority)}</span>
                    <span class="badge badge-${p.status === "VALIDATED_SUCCESS" ? "PASSED" : p.status === "ROLLED_BACK_FAILED" ? "FAILED" : "PARTIAL_SUCCESS"}">${escapeHtml(p.status)}</span>
                  </div>
                </div>
                <div style="font-size:13px; margin-top:4px;"><strong>Observed Problem:</strong> ${escapeHtml(p.observed_problem)}</div>
                <div style="font-size:12px; color:var(--text-secondary);"><strong>Likely Cause:</strong> ${escapeHtml(p.likely_cause)}</div>
                <div style="font-size:12px; color:var(--text-secondary);"><strong>Proposed Change:</strong> ${escapeHtml(p.proposed_change)}</div>
                <div class="mono" style="font-size:11px; color:var(--text-secondary);">
                  Evidence: ${escapeHtml((p.supporting_evidence || []).join(" | "))}
                </div>
                ${
                  p.status === "PROPOSED"
                    ? `<div style="margin-top:8px;">
                        <button class="btn btn-primary btn-sm" type="button" onclick="window.runControlledSelfImprovement('${escapeHtml(p.proposal_id)}')">
                          Apply &amp; Evaluate (Verification + Live Run Testing)
                        </button>
                       </div>`
                    : `<div class="mono" style="font-size:11px; color:var(--text-secondary); margin-top:4px;">Evaluation Notes: ${escapeHtml(p.evaluation_notes || "")}</div>`
                }
              </div>
            `
            )
            .join("");

    runList.innerHTML =
      runs.length === 0
        ? `<div class="empty-hint">No controlled self-improvement runs executed yet.</div>`
        : runs
            .map(
              (r) => `
              <div class="list-card">
                <div class="list-card-header">
                  <span class="mono"><strong>${escapeHtml(r.run_id)}</strong> (proposal=${escapeHtml(r.proposal_id)})</span>
                  <span class="badge badge-${r.final_disposition === "VALIDATED_SUCCESS" ? "PASSED" : "FAILED"}">${escapeHtml(r.final_disposition)}</span>
                </div>
                <div class="mono" style="font-size:12px;">
                  Verification: <strong>${escapeHtml(r.verification_status)}</strong> • Live Run Testing: <strong>${escapeHtml(r.live_test_status)}</strong> • Rolled Back: <strong>${r.rolled_back}</strong>
                </div>
                <div style="font-size:12px; color:var(--text-secondary);">${escapeHtml(r.notes)}</div>
              </div>
            `
            )
            .join("");
  } catch (err) {
    propList.innerHTML = `<div class="empty-hint">Error loading proposals: ${escapeHtml(err.message)}</div>`;
  }
}

window.runControlledSelfImprovement = async (proposalId) => {
  const banner = document.getElementById("self-improve-banner");
  if (banner) {
    banner.style.display = "block";
    banner.textContent = `Running controlled self-improvement for ${proposalId} (Verification + Live Run Testing)...`;
  }
  try {
    const res = await apiRequest("/api/self-improve", {
      method: "POST",
      body: JSON.stringify({ proposal_id: proposalId }),
    });
    const run = res.run || {};
    if (banner) {
      banner.textContent = `[${run.final_disposition}] Verification=${run.verification_status}, LiveRunTest=${run.live_test_status}, RolledBack=${run.rolled_back} — ${run.notes}`;
    }
    await loadImprovementsTab();
    await refreshSystemStatus();
  } catch (err) {
    if (banner) {
      banner.textContent = `Self-improvement error: ${err.message}`;
    }
  }
};

async function loadModelsAndProvidersTab() {
  const provContainer = document.getElementById("providers-list-container");
  const modContainer = document.getElementById("models-list-container");
  if (!provContainer || !modContainer) return;

  try {
    const [provData, modData] = await Promise.all([
      apiRequest("/api/providers"),
      apiRequest("/api/models"),
    ]);

    const providers = provData.providers || [];
    provContainer.innerHTML =
      providers.length === 0
        ? `<div class="empty-hint">No providers configured.</div>`
        : providers
            .map(
              (p) => `
              <div class="list-card">
                <div class="list-card-header">
                  <span class="mono"><strong>${escapeHtml(p.provider_id)}</strong> — ${escapeHtml(p.name)}</span>
                  <span class="badge ${p.enabled ? "badge-AVAILABLE" : "badge-FAILED"}">${p.enabled ? "ENABLED" : "DISABLED"}</span>
                </div>
                <div class="mono" style="font-size:12px; color:var(--text-secondary);">
                  Adapter: ${escapeHtml(p.adapter_type)} • Endpoint: ${escapeHtml(p.endpoint)}
                </div>
                <div class="mono" style="font-size:12px;">
                  Credentials: ${escapeHtml(p.credential_preview || "none configured")} (env: ${escapeHtml(p.api_key_env || "none")})
                </div>
                <div style="margin-top:6px; display:flex; gap:8px;">
                  <button class="btn btn-secondary btn-sm" type="button" onclick="window.toggleProviderEnabled('${escapeHtml(p.provider_id)}', ${!p.enabled})">
                    ${p.enabled ? "Disable" : "Enable"}
                  </button>
                  <button class="btn btn-secondary btn-sm" type="button" onclick="window.deleteProvider('${escapeHtml(p.provider_id)}')">
                    Remove
                  </button>
                </div>
              </div>
            `
            )
            .join("");

    const models = modData.models || [];
    modContainer.innerHTML =
      models.length === 0
        ? `<div class="empty-hint">No model profiles configured.</div>`
        : models
            .map((m) => {
              const badges = [];
              if (m.is_active) badges.push(`<span class="badge badge-PASSED">ACTIVE</span>`);
              if (m.is_default) badges.push(`<span class="badge badge-AVAILABLE">DEFAULT</span>`);
              if (m.is_fallback) badges.push(`<span class="badge badge-PARTIAL_SUCCESS">FALLBACK</span>`);
              if (!m.enabled) badges.push(`<span class="badge badge-FAILED">DISABLED</span>`);
              return `
                <div class="list-card">
                  <div class="list-card-header">
                    <span class="mono"><strong>${escapeHtml(m.profile_id)}</strong> — ${escapeHtml(m.name)}</span>
                    <div style="display:flex; gap:4px;">${badges.join("")}</div>
                  </div>
                  <div class="mono" style="font-size:12px; color:var(--text-secondary);">
                    Provider: ${escapeHtml(m.provider_id)} • Model ID: ${escapeHtml(m.model_id)} • Context Window: ${m.context_window_tokens || "unknown"}
                  </div>
                  <div class="mono" style="font-size:11px; color:var(--text-secondary);">
                    Temp: ${m.temperature} • Max Tokens: ${m.max_tokens} • Timeout: ${m.timeout_seconds}s • Retries: ${m.max_retries}
                  </div>
                  <div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:6px;">
                    <button class="btn btn-primary btn-sm" type="button" onclick="window.activateModel('${escapeHtml(m.profile_id)}', false)">
                      Activate Now
                    </button>
                    <button class="btn btn-secondary btn-sm" type="button" onclick="window.activateModel('${escapeHtml(m.profile_id)}', true)">
                      Activate &amp; Set Default
                    </button>
                    <button class="btn btn-secondary btn-sm" type="button" onclick="window.setFallbackModel('${escapeHtml(m.profile_id)}')">
                      Set Fallback
                    </button>
                    <button class="btn btn-secondary btn-sm" type="button" onclick="window.runConnectionTest('${escapeHtml(m.profile_id)}')">
                      Test Connection
                    </button>
                    <button class="btn btn-secondary btn-sm" type="button" onclick="window.deleteModel('${escapeHtml(m.profile_id)}')">
                      Remove
                    </button>
                  </div>
                </div>
              `;
            })
            .join("");
  } catch (err) {
    provContainer.innerHTML = `<div class="empty-hint">Error loading providers/models: ${escapeHtml(err.message)}</div>`;
  }
}

window.activateModel = async (profileId, setAsDefault) => {
  try {
    await apiRequest("/api/models/select", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, set_as_default: Boolean(setAsDefault) }),
    });
    await loadModelsAndProvidersTab();
    await refreshSystemStatus();
  } catch (err) {
    alert(`Model selection error: ${err.message}`);
  }
};

window.setFallbackModel = async (profileId) => {
  try {
    await apiRequest("/api/models/select", {
      method: "POST",
      body: JSON.stringify({ fallback_model: profileId }),
    });
    await loadModelsAndProvidersTab();
    await refreshSystemStatus();
  } catch (err) {
    alert(`Fallback model error: ${err.message}`);
  }
};

window.runConnectionTest = async (profileId) => {
  const banner = document.getElementById("connection-test-banner");
  if (banner) {
    banner.innerHTML = `<div class="config-status-banner">Testing connection for profile <strong>${escapeHtml(profileId)}</strong>...</div>`;
  }
  try {
    const rep = await apiRequest("/api/models/test", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    if (banner) {
      banner.innerHTML = `
        <div class="list-card" style="border-color:var(--accent);">
          <div class="list-card-header">
            <strong>Connection Test Result: ${escapeHtml(profileId)}</strong>
            <span class="badge badge-${escapeHtml(rep.status)}">${escapeHtml(rep.status)}</span>
          </div>
          <div class="mono" style="font-size:12px;">
            Provider: ${escapeHtml(rep.provider)} • Model: ${escapeHtml(rep.model_id)} • Latency: ${(rep.latency_ms || 0).toFixed(1)} ms
          </div>
          <div class="mono" style="font-size:12px; color:var(--text-secondary);">
            config_valid=${rep.config_valid} | endpoint_reachable=${rep.endpoint_reachable} | auth_valid=${rep.auth_valid} | model_accepted=${rep.model_accepted} | inference_succeeded=${rep.inference_succeeded} | conclusive=${rep.conclusive}
          </div>
          <div style="font-size:12px; margin-top:4px;">${escapeHtml(rep.details || "")}</div>
        </div>
      `;
    }
  } catch (err) {
    if (banner) {
      banner.innerHTML = `<div class="code-snippet" style="color:#ffa198;">Connection test failed: ${escapeHtml(err.message)}</div>`;
    }
  }
};

window.toggleProviderEnabled = async (providerId, enabledState) => {
  try {
    const provData = await apiRequest("/api/providers");
    const target = (provData.providers || []).find((p) => p.provider_id === providerId);
    if (!target) return;
    target.enabled = enabledState;
    await apiRequest("/api/providers", {
      method: "POST",
      body: JSON.stringify({ provider: target }),
    });
    await loadModelsAndProvidersTab();
    await refreshSystemStatus();
  } catch (err) {
    alert(`Provider update error: ${err.message}`);
  }
};

window.deleteProvider = async (providerId) => {
  try {
    await apiRequest("/api/providers/remove", {
      method: "POST",
      body: JSON.stringify({ provider_id: providerId }),
    });
    await loadModelsAndProvidersTab();
    await refreshSystemStatus();
  } catch (err) {
    alert(`Cannot remove provider: ${err.message}`);
  }
};

window.deleteModel = async (profileId) => {
  try {
    await apiRequest("/api/models/remove", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    await loadModelsAndProvidersTab();
    await refreshSystemStatus();
  } catch (err) {
    alert(`Cannot remove model: ${err.message}`);
  }
};

document.addEventListener("DOMContentLoaded", () => {
  // Navigation tabs
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
  });

  // Quick prompts in sidebar
  document.querySelectorAll(".quick-prompt-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab("chat");
      submitChatInstruction(btn.getAttribute("data-prompt"));
    });
  });

  document.getElementById("refresh-sidebar-btn").addEventListener("click", refreshSystemStatus);
  document.getElementById("reload-tasks-btn").addEventListener("click", loadTasksTab);
  document.getElementById("filter-available-caps").addEventListener("change", loadCapabilitiesTab);

  const reloadModelsBtn = document.getElementById("reload-models-tab-btn");
  if (reloadModelsBtn) {
    reloadModelsBtn.addEventListener("click", loadModelsAndProvidersTab);
  }

  const detectPropBtn = document.getElementById("detect-proposals-btn");
  if (detectPropBtn) {
    detectPropBtn.addEventListener("click", async () => {
      try {
        await apiRequest("/api/proposals", { method: "POST" });
        await loadImprovementsTab();
        await refreshSystemStatus();
      } catch (err) {
        alert(`Failed detecting proposals: ${err.message}`);
      }
    });
  }

  const refreshPropBtn = document.getElementById("refresh-proposals-btn");
  if (refreshPropBtn) {
    refreshPropBtn.addEventListener("click", loadImprovementsTab);
  }

  const provForm = document.getElementById("provider-upsert-form");
  if (provForm) {
    provForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const provider = {
          provider_id: document.getElementById("prov-id").value.trim(),
          name: document.getElementById("prov-name").value.trim(),
          adapter_type: document.getElementById("prov-adapter").value,
          endpoint: document.getElementById("prov-endpoint").value.trim(),
          api_key_env: document.getElementById("prov-env").value.trim() || "AGENT911_API_KEY",
          enabled: true,
        };
        const apiKeyVal = document.getElementById("prov-apikey").value.trim();
        await apiRequest("/api/providers", {
          method: "POST",
          body: JSON.stringify({ provider, api_key: apiKeyVal || null }),
        });
        document.getElementById("prov-apikey").value = "";
        await loadModelsAndProvidersTab();
        await refreshSystemStatus();
      } catch (err) {
        alert(`Failed saving provider: ${err.message}`);
      }
    });
  }

  const modForm = document.getElementById("model-upsert-form");
  if (modForm) {
    modForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const ctxRaw = document.getElementById("mod-ctx-window").value.trim();
        const model = {
          profile_id: document.getElementById("mod-profile-id").value.trim(),
          name: document.getElementById("mod-name").value.trim(),
          provider_id: document.getElementById("mod-provider-id").value.trim(),
          model_id: document.getElementById("mod-model-id").value.trim(),
          context_window_tokens: ctxRaw ? Number(ctxRaw) : null,
          temperature: Number(document.getElementById("mod-temperature").value || 0.1),
          timeout_seconds: Number(document.getElementById("mod-timeout").value || 30),
          enabled: true,
        };
        await apiRequest("/api/models", {
          method: "POST",
          body: JSON.stringify({ model }),
        });
        await loadModelsAndProvidersTab();
        await refreshSystemStatus();
      } catch (err) {
        alert(`Failed saving model profile: ${err.message}`);
      }
    });
  }

  // Chat composer
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const initialStreamMarkup = document.getElementById("chat-stream").innerHTML;

  function bindQuickPromptButtons(root = document) {
    root.querySelectorAll(".quick-prompt-btn").forEach((btn) => {
      if (btn.dataset.boundQuick === "1") return;
      btn.dataset.boundQuick = "1";
      btn.addEventListener("click", () => {
        const prompt = btn.getAttribute("data-prompt");
        switchTab("chat");
        submitChatInstruction(prompt);
      });
    });
  }
  bindQuickPromptButtons(document);

  const newChatBtn = document.getElementById("new-chat-btn");
  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      switchTab("chat");
      const stream = document.getElementById("chat-stream");
      stream.innerHTML = initialStreamMarkup;
      bindQuickPromptButtons(stream);
      const pop = document.getElementById("distill-preview-popover");
      if (pop) pop.classList.add("hidden");
    });
  }

  const distillPreviewBtn = document.getElementById("distill-preview-btn");
  const distillPopover = document.getElementById("distill-preview-popover");
  if (distillPreviewBtn && distillPopover) {
    distillPreviewBtn.addEventListener("click", async () => {
      let rawText = chatInput.value.trim();
      if (!rawText) {
        rawText =
          "uhh hey so like i'm trying to make a pyhton script maybe call it price_calc or something in billing/ folder... basically it needs to calculate tax and apply discount, oh and also make a config json in billing/rates.json!!";
        chatInput.value = rawText;
      }
      distillPreviewBtn.disabled = true;
      distillPreviewBtn.textContent = "✨ Distilling...";
      try {
        const spec = await apiRequest("/api/distill", {
          method: "POST",
          body: JSON.stringify({ instruction: rawText }),
        });
        const fns = (spec.extracted_functions || []).map((f) => `<code>${escapeHtml(f)}()</code>`).join(", ");
        const arts = (spec.target_artifacts || [])
          .map((a) => `<code>${escapeHtml(a.path)}</code> <span class="mono" style="opacity:0.7">(${escapeHtml(a.artifact_type)})</span>`)
          .join(", ");
        const ambs = (spec.ambiguities_resolved || [])
          .map((a) => `<li>${escapeHtml(a)}</li>`)
          .join("");
        distillPopover.innerHTML = `
          <div class="distill-popover-header">
            <div>
              <span class="distilled-badge">✨ CONCRETE BUILD SPECIFICATION</span>
              <span class="mono" style="font-size:11px; color:var(--text-secondary); margin-left:8px;">
                ${spec.is_messy_input ? "Messy Writing Distilled" : "Clean Spec"} • ${spec.is_build_intent ? "Build Intent" : "Conversational / Other"}
              </span>
            </div>
            <div style="display:flex; gap:6px;">
              <button type="button" class="btn btn-primary btn-sm" id="popover-build-now-btn">Build Now ↗</button>
              <button type="button" class="btn btn-secondary btn-sm" id="popover-close-btn">&times;</button>
            </div>
          </div>
          <div class="distill-popover-body">
            <div><strong>What You Want to Build:</strong> ${escapeHtml(spec.what_user_wants_to_build)}</div>
            ${arts ? `<div><strong>Target Artifacts:</strong> ${arts}</div>` : ""}
            ${fns ? `<div><strong>Identified Functions:</strong> ${fns}</div>` : ""}
            <div><strong>Concrete Plan:</strong> ${escapeHtml(spec.concrete_instruction)}</div>
            ${ambs ? `<ul class="distill-amb-list">${ambs}</ul>` : ""}
          </div>
        `;
        distillPopover.classList.remove("hidden");
        document.getElementById("popover-close-btn").addEventListener("click", () => {
          distillPopover.classList.add("hidden");
        });
        document.getElementById("popover-build-now-btn").addEventListener("click", () => {
          distillPopover.classList.add("hidden");
          submitChatInstruction(chatInput.value);
        });
      } catch (err) {
        alert(`Distill preview error: ${err.message}`);
      } finally {
        distillPreviewBtn.disabled = false;
        distillPreviewBtn.textContent = "✨ Distill Preview";
      }
    });
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitChatInstruction(chatInput.value);
  });
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitChatInstruction(chatInput.value);
    }
  });

  // Grow capability form
  document.getElementById("grow-capability-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("grow-cap-submit");
    const resultBox = document.getElementById("grow-capability-result");
    btn.disabled = true;
    resultBox.innerHTML = `<div class="empty-hint">Developing, verifying, and live-testing capability...</div>`;

    try {
      const task = await apiRequest("/api/capabilities/develop", {
        method: "POST",
        body: JSON.stringify({
          capability_id: document.getElementById("grow-cap-id").value.trim(),
          name: document.getElementById("grow-cap-name").value.trim(),
          purpose: document.getElementById("grow-cap-purpose").value.trim(),
          implementation_code: document.getElementById("grow-cap-code").value,
        }),
      });
      resultBox.innerHTML = renderTaskCard(task, false);
      await loadCapabilitiesTab();
      await refreshSystemStatus();
    } catch (err) {
      resultBox.innerHTML = `<div class="code-snippet" style="border-color:var(--danger); color:#ffa198;">${escapeHtml(err.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  });

  // Memory search & store forms
  document.getElementById("memory-search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadMemoriesAndExperiences();
  });
  document.getElementById("experience-search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadMemoriesAndExperiences();
  });
  document.getElementById("memory-store-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await apiRequest("/api/memory", {
        method: "POST",
        body: JSON.stringify({
          category: document.getElementById("store-mem-category").value,
          confidence: document.getElementById("store-mem-confidence").value,
          key: document.getElementById("store-mem-key").value.trim(),
          content: document.getElementById("store-mem-content").value.trim(),
        }),
      });
      document.getElementById("store-mem-key").value = "";
      document.getElementById("store-mem-content").value = "";
      await loadMemoriesAndExperiences();
      await refreshSystemStatus();
    } catch (err) {
      alert(`Failed storing memory: ${err.message}`);
    }
  });

  // Model & Config Modal
  const modal = document.getElementById("config-modal");
  const openModal = () => {
    if (cachedStatus && cachedStatus.configuration) {
      const cfg = cachedStatus.configuration;
      const m = cfg.model || {};
      const conn = cachedStatus.model_connectivity || {};
      document.getElementById("cfg-provider").value = m.provider || "openai_compatible";
      document.getElementById("cfg-model-id").value = m.model_id || "gpt-4o-mini";
      document.getElementById("cfg-endpoint").value =
        m.endpoint || "https://api.openai.com/v1/chat/completions";
      document.getElementById("cfg-timeout").value = m.timeout_seconds || 30;
      document.getElementById("cfg-temperature").value =
        m.temperature !== undefined ? m.temperature : 0.1;
      document.getElementById("cfg-max-recovery").value =
        cfg.max_recovery_attempts_per_step !== undefined
          ? cfg.max_recovery_attempts_per_step
          : 2;
      document.getElementById("cfg-local-fallback").checked =
        m.allow_local_fallback !== false;
      document.getElementById("modal-connectivity-banner").textContent =
        `Current Connectivity: ${conn.status || "UNKNOWN"} — ${conn.details || ""}`;
    }
    modal.classList.remove("hidden");
  };
  const closeModal = () => modal.classList.add("hidden");

  document.getElementById("open-config-btn").addEventListener("click", openModal);
  document.getElementById("close-config-btn").addEventListener("click", closeModal);
  document.getElementById("cancel-config-btn").addEventListener("click", closeModal);

  document.getElementById("config-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await apiRequest("/api/config", {
        method: "POST",
        body: JSON.stringify({
          max_recovery_attempts_per_step: Number(
            document.getElementById("cfg-max-recovery").value
          ),
          model: {
            provider: document.getElementById("cfg-provider").value,
            model_id: document.getElementById("cfg-model-id").value.trim(),
            endpoint: document.getElementById("cfg-endpoint").value.trim(),
            timeout_seconds: Number(document.getElementById("cfg-timeout").value),
            temperature: Number(document.getElementById("cfg-temperature").value),
            allow_local_fallback: document.getElementById("cfg-local-fallback").checked,
          },
        }),
      });
      await refreshSystemStatus();
      closeModal();
    } catch (err) {
      alert(`Configuration error: ${err.message}`);
    }
  });

  refreshSystemStatus();
});
