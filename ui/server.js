/**
 * Agent 911 Node.js Server & Chat Interface Backend.
 * Binds to 0.0.0.0, supports iframe preview origins, and bridges REST requests
 * directly to the Python Agent911Orchestrator (`python3 -m agent911.bridge`).
 */

"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const REPO_ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

/**
 * Invoke `python3 -m agent911.bridge` with a JSON request payload.
 */
function callPythonBridge(payload, options = {}) {
  const pythonBin = options.pythonBin || process.env.PYTHON_BIN || "python3";
  const workspace =
    payload.workspace ||
    options.workspace ||
    process.env.AGENT911_WORKSPACE ||
    REPO_ROOT;

  const fullPayload = { ...payload, workspace };

  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, ["-m", "agent911.bridge"], {
      cwd: REPO_ROOT,
      env: { ...process.env, PYTHONPATH: REPO_ROOT },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Python bridge timed out after 60s."));
    }, 60000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });

    child.on("close", () => {
      clearTimeout(timer);
      const trimmed = stdout.trim();
      if (!trimmed) {
        return reject(
          new Error(`Bridge returned empty output. stderr: ${stderr.trim()}`)
        );
      }
      try {
        const parsed = JSON.parse(trimmed);
        resolve(parsed);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse bridge JSON output: ${err.message}. Raw: ${trimmed.slice(0, 300)}`
          )
        );
      }
    });

    child.stdin.write(JSON.stringify(fullPayload));
    child.stdin.end();
  });
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk.toString("utf8");
      if (raw.length > 2_000_000) {
        reject(new Error("Request payload too large."));
      }
    });
    req.on("end", () => {
      if (!raw.trim()) {
        return resolve({});
      }
      try {
        resolve(JSON.parse(raw));
      } catch (err) {
        reject(new Error(`Invalid JSON body: ${err.message}`));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, statusCode, data) {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function createAppServer(options = {}) {
  const server = http.createServer(async (req, res) => {
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      });
      return res.end();
    }

    const parsedUrl = new URL(req.url || "/", "http://localhost");
    const pathname = parsedUrl.pathname;

    try {
      // API Routes
      if (pathname === "/api/health" && req.method === "GET") {
        return sendJson(res, 200, { ok: true, service: "agent911-node-ui" });
      }

      if (pathname === "/api/status" && req.method === "GET") {
        const result = await callPythonBridge({ action: "status" }, options);
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/config" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          { action: "update_config", updates: body },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/providers" && req.method === "GET") {
        const result = await callPythonBridge(
          { action: "list_providers" },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/providers" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "upsert_provider",
            provider: body.provider || body,
            api_key: body.api_key || null,
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/providers/remove" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          { action: "remove_provider", provider_id: body.provider_id },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/models" && req.method === "GET") {
        const result = await callPythonBridge(
          { action: "list_models" },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/models" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          { action: "upsert_model", model: body.model || body },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/models/remove" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          { action: "remove_model", profile_id: body.profile_id },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/models/select" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "select_model",
            profile_id: body.profile_id,
            set_as_default: Boolean(body.set_as_default),
            ...(Object.prototype.hasOwnProperty.call(body, "fallback_model")
              ? { fallback_model: body.fallback_model }
              : {}),
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/models/test" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "test_model_connection",
            profile_id: body.profile_id || null,
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/chat" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "chat",
            instruction: body.instruction,
            explicit_plan: body.explicit_plan,
            explicit_requirements: body.explicit_requirements,
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/tasks" && req.method === "GET") {
        const limit = Number(parsedUrl.searchParams.get("limit") || 30);
        const result = await callPythonBridge(
          { action: "list_tasks", limit },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      const taskResumeMatch = pathname.match(
        /^\/api\/tasks\/([^/]+)\/resume$/
      );
      if (taskResumeMatch && req.method === "POST") {
        const taskId = decodeURIComponent(taskResumeMatch[1]);
        const result = await callPythonBridge(
          { action: "resume_task", task_id: taskId },
          options
        );
        return sendJson(res, result.ok ? 200 : 404, result);
      }

      const taskInspectMatch = pathname.match(/^\/api\/tasks\/([^/]+)$/);
      if (taskInspectMatch && req.method === "GET") {
        const taskId = decodeURIComponent(taskInspectMatch[1]);
        const result = await callPythonBridge(
          { action: "inspect_task", task_id: taskId },
          options
        );
        return sendJson(res, result.ok ? 200 : 404, result);
      }

      if (pathname === "/api/capabilities" && req.method === "GET") {
        const onlyAvail =
          parsedUrl.searchParams.get("only_available") === "true";
        const result = await callPythonBridge(
          { action: "list_capabilities", only_available: onlyAvail },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/capabilities/develop" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "develop_capability",
            capability_id: body.capability_id,
            name: body.name,
            purpose: body.purpose,
            implementation_code: body.implementation_code,
            overwrite: body.overwrite !== false,
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/memory" && req.method === "GET") {
        const query = parsedUrl.searchParams.get("query") || null;
        const category = parsedUrl.searchParams.get("category") || null;
        const result = await callPythonBridge(
          { action: "list_memories", query, category },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/memory" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          {
            action: "store_memory",
            key: body.key,
            content: body.content,
            category: body.category || "FACTUAL",
            confidence: body.confidence || "OBSERVED",
          },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/experiences" && req.method === "GET") {
        const query = parsedUrl.searchParams.get("query") || null;
        const result = await callPythonBridge(
          { action: "list_experiences", query },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/proposals" && req.method === "GET") {
        const status = parsedUrl.searchParams.get("status") || null;
        const result = await callPythonBridge(
          { action: "list_proposals", status },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/proposals" && req.method === "POST") {
        const result = await callPythonBridge(
          { action: "detect_proposals" },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/self-improve" && req.method === "POST") {
        const body = await readJsonBody(req);
        const result = await callPythonBridge(
          { action: "self_improve", proposal_id: body.proposal_id },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname === "/api/diagnose" && req.method === "GET") {
        const taskId = parsedUrl.searchParams.get("task_id") || "";
        const result = await callPythonBridge(
          { action: "diagnose_task", task_id: taskId },
          options
        );
        return sendJson(res, result.ok ? 200 : 400, result);
      }

      if (pathname.startsWith("/api/")) {
        return sendJson(res, 404, {
          ok: false,
          error: `Unknown API route: ${req.method} ${pathname}`,
        });
      }

      // Static Assets
      const relFile = pathname === "/" ? "index.html" : pathname.slice(1);
      const safePath = path.resolve(PUBLIC_DIR, relFile);
      if (!safePath.startsWith(PUBLIC_DIR)) {
        res.writeHead(403);
        return res.end("Forbidden");
      }

      if (fs.existsSync(safePath) && fs.statSync(safePath).isFile()) {
        const ext = path.extname(safePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || "application/octet-stream";
        res.writeHead(200, {
          "Content-Type": contentType,
          "Access-Control-Allow-Origin": "*",
        });
        return fs.createReadStream(safePath).pipe(res);
      }

      // Fallback to index.html for SPA navigation
      const indexFile = path.join(PUBLIC_DIR, "index.html");
      res.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
      });
      return fs.createReadStream(indexFile).pipe(res);
    } catch (err) {
      return sendJson(res, 500, {
        ok: false,
        error: err.message || String(err),
        error_type: "ServerError",
      });
    }
  });

  return server;
}

if (require.main === module) {
  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || "0.0.0.0";
  const server = createAppServer();
  server.listen(port, host, () => {
    console.log(`Agent 911 Node Chat UI listening on http://${host}:${port}`);
  });
}

module.exports = { createAppServer, callPythonBridge };
