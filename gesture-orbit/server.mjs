import { appendFileSync, existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";

const root = import.meta.dirname;
const preferredPort = Number(process.env.PORT || 5197);
const logPath = join(root, "server.log");

const types = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

function log(message) {
  appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
}

process.on("uncaughtException", (error) => {
  log(`uncaughtException: ${error.stack || error.message}`);
});

function createStaticServer() {
  return createServer((request, response) => {
    try {
      const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
      const requestPath = decodeURIComponent(url.pathname);
      const relativePath =
        requestPath === "/"
          ? "index.html"
          : normalize(requestPath.replace(/^[/\\]+/, "")).replace(/^(\.\.[/\\])+/, "");
      const filePath = join(root, relativePath);

      if (!filePath.startsWith(root) || !existsSync(filePath) || !statSync(filePath).isFile()) {
        response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
        response.end("Not found");
        return;
      }

      const body = readFileSync(filePath);
      response.writeHead(200, {
        "Content-Type": types[extname(filePath)] || "application/octet-stream",
        "Content-Length": body.length,
      });

      if (request.method === "HEAD") {
        response.end();
        return;
      }

      response.end(body);
    } catch (error) {
      log(`request error: ${error.stack || error.message}`);
      if (!response.headersSent) {
        response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      }
      response.end("Server error");
    }
  });
}

function listen(port, attempts = 0) {
  const server = createStaticServer();

  server.once("error", (error) => {
    if (error.code === "EADDRINUSE" && attempts < 20) {
      listen(port + 1, attempts + 1);
      return;
    }

    log(`listen error: ${error.stack || error.message}`);
  });

  server.listen(port, "127.0.0.1", () => {
    console.log(`Gesture Orbit running at http://127.0.0.1:${port}`);
  });
}

listen(preferredPort);
