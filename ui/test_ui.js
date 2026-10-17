"use strict";

const { describe, it, before, after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { createAppServer } = require("./server");

describe("Agent 911 Node.js Chat Interface Server", () => {
  let tmpWorkspace;
  let server;
  let baseUrl;

  before(async () => {
    tmpWorkspace = fs.mkdtempSync(path.join(os.tmpdir(), "agent911-node-ui-"));
    server = createAppServer({ workspace: tmpWorkspace });
    await new Promise((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const addr = server.address();
    baseUrl = `http://127.0.0.1:${addr.port}`;
  });

  after(async () => {
    if (server) {
      await new Promise((resolve) => server.close(() => resolve()));
    }
    if (tmpWorkspace && fs.existsSync(tmpWorkspace)) {
      fs.rmSync(tmpWorkspace, { recursive: true, force: true });
    }
  });

  it("serves static HTML, CSS, and JS assets", async () => {
    const htmlRes = await fetch(`${baseUrl}/`);
    assert.equal(htmlRes.status, 200);
    const html = await htmlRes.text();
    assert.ok(html.includes("AGENT 911"));

    const cssRes = await fetch(`${baseUrl}/styles.css`);
    assert.equal(cssRes.status, 200);

    const jsRes = await fetch(`${baseUrl}/app.js`);
    assert.equal(jsRes.status, 200);
  });

  it("returns real application status from /api/status", async () => {
    const res = await fetch(`${baseUrl}/api/status`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
    assert.ok(body.data.capabilities_registered >= 18);
    assert.equal(typeof body.data.model_connectivity.status, "string");
  });

  it("executes a chat instruction through /api/chat and returns Verification and Live Run Testing results", async () => {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction: "Write a Python file 'ui_helper.py' with a function 'format_label'.",
      }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
    assert.equal(body.data.final_outcome, "SUCCESS");
    assert.ok(body.data.verification_results.length >= 3);
    assert.ok(body.data.live_run_results.length >= 1);
    assert.ok(fs.existsSync(path.join(tmpWorkspace, "ui_helper.py")));

    // Inspect the task via /api/tasks/:id
    const taskId = body.data.task_id;
    const inspRes = await fetch(`${baseUrl}/api/tasks/${encodeURIComponent(taskId)}`);
    assert.equal(inspRes.status, 200);
    const inspBody = await inspRes.json();
    assert.equal(inspBody.ok, true);
    assert.equal(inspBody.data.task_id, taskId);
    assert.ok(Array.isArray(inspBody.data.execution_logs));
  });

  it("stores and retrieves SQLite memories via /api/memory and experiences via /api/experiences", async () => {
    const storeRes = await fetch(`${baseUrl}/api/memory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category: "FACTUAL",
        confidence: "VERIFIED",
        key: "ui.framework",
        content: "Node.js native HTTP server with Python bridge",
      }),
    });
    assert.equal(storeRes.status, 200);
    const storeBody = await storeRes.json();
    assert.equal(storeBody.ok, true);

    const listRes = await fetch(`${baseUrl}/api/memory?query=bridge`);
    assert.equal(listRes.status, 200);
    const listBody = await listRes.json();
    assert.equal(listBody.ok, true);
    assert.equal(listBody.data.count, 1);
    assert.equal(listBody.data.records[0].key, "ui.framework");

    const expRes = await fetch(`${baseUrl}/api/experiences`);
    assert.equal(expRes.status, 200);
    const expBody = await expRes.json();
    assert.equal(expBody.ok, true);
    assert.ok(expBody.data.count >= 1);
  });

  it("develops and registers a new capability via /api/capabilities/develop", async () => {
    const devRes = await fetch(`${baseUrl}/api/capabilities/develop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        capability_id: "hash.sha256",
        name: "SHA256 Checksum Calculator",
        purpose: "Compute sha256 hash of input text or file",
      }),
    });
    assert.equal(devRes.status, 200);
    const devBody = await devRes.json();
    assert.equal(devBody.ok, true);
    assert.equal(devBody.data.final_outcome, "SUCCESS");

    const capsRes = await fetch(`${baseUrl}/api/capabilities`);
    const capsBody = await capsRes.json();
    assert.equal(capsBody.ok, true);
    const found = capsBody.data.capabilities.find(
      (c) => c.identifier === "hash.sha256"
    );
    assert.ok(found);
    assert.equal(found.availability, "AVAILABLE");
    assert.equal(found.is_custom_grown, true);
  });

  it("manages providers, model profiles, active selection, and connection testing via /api/providers and /api/models", async () => {
    // 1. Add a provider with an API key stored in the mode-0600 CredentialStore
    const provRes = await fetch(`${baseUrl}/api/providers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: {
          provider_id: "anthropic_ui_test",
          name: "Anthropic UI Provider",
          adapter_type: "anthropic",
          endpoint: "https://api.anthropic.com/v1/messages",
          api_key_env: "ANTHROPIC_API_KEY",
          enabled: true,
        },
        api_key: "sk-ant-ui-test-key-99999999",
      }),
    });
    assert.equal(provRes.status, 200);
    const provBody = await provRes.json();
    assert.equal(provBody.ok, true);
    const addedProv = provBody.data.providers.find(
      (p) => p.provider_id === "anthropic_ui_test"
    );
    assert.ok(addedProv);
    assert.equal(addedProv.has_credentials, true);
    assert.ok(!addedProv.credential_preview.includes("99999999"));

    // 2. Add a model profile under anthropic_ui_test
    const modRes = await fetch(`${baseUrl}/api/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: {
          profile_id: "claude-ui-profile",
          name: "Claude UI Profile",
          provider_id: "anthropic_ui_test",
          model_id: "claude-3-5-sonnet-latest",
          context_window_tokens: 200000,
          temperature: 0.2,
          timeout_seconds: 25,
          enabled: true,
        },
      }),
    });
    assert.equal(modRes.status, 200);
    const modBody = await modRes.json();
    assert.equal(modBody.ok, true);

    // 3. Switch active model and set fallback model
    const selRes = await fetch(`${baseUrl}/api/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile_id: "claude-ui-profile",
        set_as_default: true,
        fallback_model: "local-planner",
      }),
    });
    assert.equal(selRes.status, 200);
    const selBody = await selRes.json();
    assert.equal(selBody.ok, true);
    assert.equal(selBody.data.configuration.active_model, "claude-ui-profile");
    assert.equal(selBody.data.configuration.default_model, "claude-ui-profile");
    assert.equal(selBody.data.configuration.fallback_model, "local-planner");

    // 4. Test connection on local-planner profile
    const testRes = await fetch(`${baseUrl}/api/models/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: "local-planner" }),
    });
    assert.equal(testRes.status, 200);
    const testBody = await testRes.json();
    assert.equal(testBody.ok, true);
    assert.equal(testBody.data.status, "LOCAL_ONLY");
    assert.equal(testBody.data.config_valid, true);
  });

  it("detects improvement proposals, executes controlled self-improvement, and returns task diagnostics via /api/proposals, /api/self-improve, and /api/diagnose", async () => {
    // 1. Trigger a task with a recoverable ParentDirectoryNotFoundError
    const chatRes = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction: "Write nested file 'ui_impr_dir/test.txt'",
        explicit_plan: [
          {
            step_id: "step-1",
            description: "Write ui_impr_dir/test.txt without create_parents",
            capability_id: "fs.write",
            parameters: {
              path: "ui_impr_dir/test.txt",
              content: "hello from ui test\n",
              create_parents: false,
            },
          },
        ],
      }),
    });
    assert.equal(chatRes.status, 200);
    const chatBody = await chatRes.json();
    assert.equal(chatBody.ok, true);
    const taskId = chatBody.data.task_id;

    // 2. Inspect structured diagnostics for that task
    const diagRes = await fetch(`${baseUrl}/api/diagnose?task_id=${encodeURIComponent(taskId)}`);
    assert.equal(diagRes.status, 200);
    const diagBody = await diagRes.json();
    assert.equal(diagBody.ok, true);
    assert.equal(diagBody.data.task_id, taskId);
    assert.ok(diagBody.data.diagnostic_reports.length >= 1);

    // 3. Detect improvement proposals via POST /api/proposals
    const propRes = await fetch(`${baseUrl}/api/proposals`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(propRes.status, 200);
    const propBody = await propRes.json();
    assert.equal(propBody.ok, true);
    assert.ok(propBody.data.proposals.length >= 1);

    const targetProposal = propBody.data.proposals[0];

    // 4. Execute controlled self-improvement via POST /api/self-improve
    const imprRes = await fetch(`${baseUrl}/api/self-improve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proposal_id: targetProposal.proposal_id }),
    });
    assert.equal(imprRes.status, 200);
    const imprBody = await imprRes.json();
    assert.equal(imprBody.ok, true);
    assert.equal(imprBody.data.run.final_disposition, "VALIDATED_SUCCESS");
    assert.equal(imprBody.data.run.verification_status, "PASSED");
    assert.equal(imprBody.data.run.live_test_status, "PASSED");
  });

  it("supports both natural conversational chatting and autonomous instruction execution via /api/chat", async () => {
    // 1. Conversational turn: introduce user
    const r1 = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: "Hello! My name is Aryan" }),
    });
    assert.equal(r1.status, 200);
    const b1 = await r1.json();
    assert.equal(b1.ok, true);
    assert.equal(b1.data.interaction_mode, "conversational");
    assert.ok(b1.data.assistant_response.includes("Aryan"));

    // 2. Conversational recall turn: ask "What is my name?"
    const r2 = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: "What is my name?" }),
    });
    assert.equal(r2.status, 200);
    const b2 = await r2.json();
    assert.equal(b2.ok, true);
    assert.equal(b2.data.interaction_mode, "conversational");
    assert.ok(b2.data.assistant_response.includes("Aryan"));

    // 3. Autonomous task instruction turn: inspect workspace
    const r3 = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: "Inspect project structure" }),
    });
    assert.equal(r3.status, 200);
    const b3 = await r3.json();
    assert.equal(b3.ok, true);
    assert.equal(b3.data.interaction_mode, "task_execution");
    assert.ok(b3.data.assistant_response.includes("project files"));

    // 4. Distill and build from messy user writing via /api/distill and /api/chat
    const messyWriting =
      "uhh hey so like i'm trying to make a pyhton script maybe call it price_calc or something in billing/ folder... basically it needs to calculate tax and apply discount, oh and also make a config json in billing/rates.json!!";
    const rDistill = await fetch(`${baseUrl}/api/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: messyWriting }),
    });
    assert.equal(rDistill.status, 200);
    const bDistill = await rDistill.json();
    assert.equal(bDistill.ok, true);
    assert.equal(bDistill.data.is_messy_input, true);
    assert.equal(bDistill.data.is_build_intent, true);
    assert.ok(bDistill.data.what_user_wants_to_build.includes("billing/price_calc.py"));

    const rBuild = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: messyWriting }),
    });
    assert.equal(rBuild.status, 200);
    const bBuild = await rBuild.json();
    assert.equal(bBuild.ok, true);
    assert.equal(bBuild.data.final_outcome, "SUCCESS");
    assert.equal(bBuild.data.distilled_spec.is_messy_input, true);
    assert.ok(bBuild.data.assistant_response.includes("Distilled Build Objective"));
  });
});
