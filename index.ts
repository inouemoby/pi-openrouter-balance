import type { AssistantMessage } from "@earendil-works/pi-ai";
import { type ExtensionAPI, readStoredCredential } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import * as fs from "node:fs";
import * as path from "node:path";

const PROVIDER = "openrouter";
const CREDITS_URL = "https://openrouter.ai/api/v1/credits";
const MODELS_URL = "https://openrouter.ai/api/v1/models";
const MODELS_CONFIG_PATH = "models.json";
const CACHE_MS = 60_000;
const IDLE_REFRESH_MS = 5 * 60 * 1000;

interface BalanceData {
  totalCredits: number;
  totalUsage: number;
  remaining: number;
  _ts: number;
}

function getAuthPath(): string {
  const agentDir = process.env.PI_CODING_AGENT_DIR
    || path.join(process.env.USERPROFILE || process.env.HOME || ".", ".pi", "agent");
  return path.join(agentDir, "auth.json");
}

function readApiKey(): string {
  if (process.env.OPENROUTER_API_KEY?.trim()) return process.env.OPENROUTER_API_KEY.trim();

  try {
    const credential = readStoredCredential(PROVIDER);
    if (credential?.type === "api_key" && credential.key) return credential.key;
    if (credential?.type === "oauth" && credential.access) return credential.access;
  } catch {
    // Fall through to the direct auth-file fallback.
  }

  try {
    const auth = JSON.parse(fs.readFileSync(getAuthPath(), "utf8"));
    const credential = auth[PROVIDER];
    if (credential?.type === "api_key" && typeof credential.key === "string") return credential.key;
    if (credential?.type === "oauth" && typeof credential.access === "string") return credential.access;
  } catch {
    // No configured OpenRouter credential.
  }
  return "";
}

function money(value: number): string {
  return `$${Math.max(0, value).toFixed(4)}`;
}

function formatTokens(count: number): string {
  if (count < 1000) return String(count);
  if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
  if (count < 1e6) return `${Math.round(count / 1000)}k`;
  if (count < 1e7) return `${(count / 1e6).toFixed(1)}M`;
  return `${Math.round(count / 1e6)}M`;
}

function getModelsConfigPath(): string {
  const agentDir = process.env.PI_CODING_AGENT_DIR
    || path.join(process.env.USERPROFILE || process.env.HOME || ".", ".pi", "agent");
  return path.join(agentDir, MODELS_CONFIG_PATH);
}

function readModelsConfig(): Record<string, any> {
  try {
    const file = getModelsConfigPath();
    return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  } catch {
    return {};
  }
}

function writeModelsConfig(config: Record<string, any>): void {
  const file = getModelsConfigPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(config, null, 2) + "\n", "utf8");
  fs.renameSync(temporary, file);
}

async function setAllOpenRouterFlex(enabled: boolean): Promise<number> {
  const config = readModelsConfig();
  config.providers ??= {};
  config.providers.openrouter ??= {};
  const overrides = config.providers.openrouter.modelOverrides ?? {};

  if (enabled) {
    const key = readApiKey();
    if (!key) throw new Error("No OpenRouter API key found.");
    const response = await fetch(MODELS_URL, {
      headers: { Accept: "application/json", Authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) throw new Error(`OpenRouter models request failed: HTTP ${response.status}`);
    const body: any = await response.json();
    const ids = (body?.data || [])
      .map((model: any) => typeof model?.id === "string" ? model.id : "")
      .filter((id: string) => id && !id.includes(":"));
    if (!ids.length) throw new Error("OpenRouter returned no model IDs.");
    for (const id of ids) {
      const override = overrides[id] ?? {};
      override.samplingParams = { ...(override.samplingParams ?? {}), service_tier: "flex" };
      // Gemini encrypted reasoning signatures are tied to the upstream route.
      // Keep Gemini 3.8 on one Flex provider instead of alternating between
      // Google AI Studio and Vertex across turns.
      if (id === "google/gemini-3.8-flash") {
        override.compat = {
          ...(override.compat ?? {}),
          openRouterRouting: {
            ...(override.compat?.openRouterRouting ?? {}),
            only: ["google-ai-studio/flex"],
          },
        };
      }
      overrides[id] = override;
    }
  } else {
    for (const override of Object.values(overrides) as any[]) {
      if (!override?.samplingParams) continue;
      delete override.samplingParams.service_tier;
      if (!Object.keys(override.samplingParams).length) delete override.samplingParams;
      if (override.compat?.openRouterRouting?.only?.length === 1
        && override.compat.openRouterRouting.only[0] === "google-ai-studio/flex") {
        delete override.compat.openRouterRouting.only;
        if (!Object.keys(override.compat.openRouterRouting).length) delete override.compat.openRouterRouting;
        if (!Object.keys(override.compat).length) delete override.compat;
      }
    }
  }

  config.providers.openrouter.modelOverrides = overrides;
  writeModelsConfig(config);
  return Object.values(overrides).filter((override: any) => override?.samplingParams?.service_tier === "flex").length;
}

function countFlexOverrides(): number {
  const overrides = readModelsConfig().providers?.openrouter?.modelOverrides ?? {};
  return Object.values(overrides).filter((override: any) => override?.samplingParams?.service_tier === "flex").length;
}

function stripOpenRouterReasoningSignatures(payload: unknown): unknown {
  if (!payload || typeof payload !== "object" || !Array.isArray((payload as any).messages)) return payload;
  return {
    ...(payload as Record<string, unknown>),
    messages: (payload as any).messages.map((message: any) => {
      if (!message || message.role !== "assistant") return message;
      const clean = { ...message };
      // These fields can contain encrypted/provider-specific reasoning from a
      // previous route. Keep visible text and tool calls, but force OpenRouter
      // to start a fresh reasoning chain on the recovery request.
      delete clean.reasoning_details;
      delete clean.reasoning;
      delete clean.reasoning_content;
      delete clean.reasoning_text;
      return clean;
    }),
  };
}

function hasInvalidThoughtSignature(messages: any[], entries: any[] = []): boolean {
  const lastPersisted = entries.at(-1)?.message;
  const candidates = [
    ...messages,
    ...(lastPersisted ? [lastPersisted] : []),
  ];
  return candidates.some((message) => {
    if (!message || message.role !== "assistant") return false;
    const text = [message.errorMessage, message.content, message.message]
      .filter((value) => typeof value === "string")
      .join(" ");
    return /invalid\s+(?:thought|thinking)\s+signature/i.test(text);
  });
}

async function fetchBalance(): Promise<BalanceData> {
  const key = readApiKey();
  if (!key) throw new Error("No OpenRouter API key found. Run /login openrouter or set OPENROUTER_API_KEY.");

  const response = await fetch(CREDITS_URL, {
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${key}`,
      "User-Agent": "pi-openrouter-balance",
    },
    signal: AbortSignal.timeout(10_000),
  });
  const text = await response.text();
  let body: any;
  try { body = JSON.parse(text); } catch { body = undefined; }
  if (!response.ok) {
    const message = body?.error?.message || body?.message || text || `HTTP ${response.status}`;
    throw new Error(`OpenRouter credits request failed: ${message}`);
  }

  const totalCredits = Number(body?.data?.total_credits);
  const totalUsage = Number(body?.data?.total_usage);
  if (!Number.isFinite(totalCredits) || !Number.isFinite(totalUsage)) {
    throw new Error("OpenRouter returned an invalid credits response.");
  }
  return {
    totalCredits,
    totalUsage,
    remaining: totalCredits - totalUsage,
    _ts: Date.now(),
  };
}

export default function piOpenRouterBalance(pi: ExtensionAPI): void {
  let balance: BalanceData | null = null;
  let request: Promise<BalanceData> | null = null;
  let footerOn = false;
  let tui: any = null;
  let latestCtx: any = null;
  let busy = false;
  let timer: ReturnType<typeof setInterval> | null = null;
  let recoveryPending = false;
  let recoveryAttempts = 0;
  const MAX_RECOVERY_ATTEMPTS = 3;

  function isOpenRouter(ctx: any): boolean {
    return ctx?.model?.provider === PROVIDER;
  }

  function trigger(): void {
    if (tui) setTimeout(() => tui.requestRender?.(), 0);
  }

  async function getBalance(force = false): Promise<BalanceData> {
    if (!force && balance && Date.now() - balance._ts < CACHE_MS) return balance;
    if (request) return request;
    request = fetchBalance()
      .then((value) => {
        balance = value;
        return value;
      })
      .finally(() => { request = null; });
    return request;
  }

  async function refresh(ctx: any, force = false): Promise<void> {
    if (!isOpenRouter(ctx)) {
      if (balance) {
        balance = null;
        toggleFooter(ctx);
      }
      return;
    }
    try {
      await getBalance(force);
      trigger();
    } catch {
      // Keep the footer quiet when the service is temporarily unavailable.
    }
  }

  function toggleFooter(ctx: any): void {
    if (isOpenRouter(ctx) && readApiKey()) {
      if (!footerOn) {
        ctx.ui.setFooter(buildFooter(ctx));
        footerOn = true;
      }
    } else if (footerOn) {
      tui = null;
      ctx.ui.setFooter(undefined as any);
      footerOn = false;
    }
  }

  function buildFooter(ctx: any) {
    return (nextTui: any, theme: any, fd: any) => {
      tui = nextTui;
      const unsubscribe = fd.onBranchChange(() => nextTui.requestRender());
      return {
        dispose: () => {
          unsubscribe();
          tui = null;
          footerOn = false;
        },
        invalidate() {},
        render(width: number): string[] {
          const sm = ctx.sessionManager;
          const home = process.env.HOME || process.env.USERPROFILE || "";
          let cwd = ctx.cwd || sm?.getCwd?.() || "";
          if (home && cwd.startsWith(home)) cwd = "~" + cwd.slice(home.length);
          const branch = fd?.getGitBranch?.();
          if (branch) cwd += ` (${branch})`;
          const name = sm?.getSessionName?.();
          if (name) cwd += ` • ${name}`;
          const line1 = truncateToWidth(theme.fg("dim", cwd), width, theme.fg("dim", "..."));

          let input = 0, output = 0, cacheRead = 0, cacheWrite = 0, cost = 0;
          for (const entry of sm?.getEntries?.() || []) {
            if (entry.type !== "message" || entry.message?.role !== "assistant") continue;
            const usage = (entry.message as AssistantMessage).usage;
            if (!usage) continue;
            input += usage.input || 0;
            output += usage.output || 0;
            cacheRead += usage.cacheRead || 0;
            cacheWrite += usage.cacheWrite || 0;
            cost += usage.cost?.total || 0;
          }

          const parts: string[] = [];
          const removable: number[] = [];
          if (input) parts.push(`↑${formatTokens(input)}`);
          if (output) parts.push(`↓${formatTokens(output)}`);
          if (cacheRead) { removable.push(parts.length); parts.push(`R${formatTokens(cacheRead)}`); }
          if (cacheWrite) { removable.push(parts.length); parts.push(`W${formatTokens(cacheWrite)}`); }
          if (cost) parts.push(money(cost));

          const contextUsage = ctx.getContextUsage?.();
          const contextWindow = contextUsage?.contextWindow ?? ctx.model?.contextWindow ?? 0;
          const contextPercent = contextUsage?.percent;
          const contextText = contextPercent === null || contextPercent === undefined
            ? `?/${formatTokens(contextWindow)} (auto)`
            : `${contextPercent.toFixed(1)}%/${formatTokens(contextWindow)} (auto)`;
          parts.push(contextText);

          if (balance) {
            const balanceText = `OR:${money(balance.remaining)}`;
            const index = parts.length;
            parts.push(balanceText);
            const left = parts.join(" ");
            if (balance.remaining <= 0) parts[index] = theme.fg("error", balanceText);
            else if (balance.remaining < 1) parts[index] = theme.fg("warning", balanceText);
          }

          let left = parts.join(" ");
          const model = ctx.model;
          let right = model?.id || "no-model";
          if (model?.reasoning) {
            const level = pi.getThinkingLevel?.() || "off";
            right += level === "off" ? " • thinking off" : ` • ${level}`;
          }
          const providerRight = model ? `(${model.provider}) ${right}` : right;
          if (visibleWidth(left) + 2 + visibleWidth(providerRight) <= width) right = providerRight;

          const fits = () => visibleWidth(left) + 2 + visibleWidth(right) <= width;
          if (!fits()) {
            for (const index of removable) parts[index] = "";
            left = parts.filter(Boolean).join(" ");
          }

          const leftWidth = visibleWidth(left);
          const rightWidth = visibleWidth(right);
          const line2 = leftWidth + 2 + rightWidth <= width
            ? left + " ".repeat(width - leftWidth - rightWidth) + right
            : truncateToWidth(left + "  " + right, width, "");
          return [line1, theme.fg("dim", line2)];
        },
      };
    };
  }

  function stopTimer(): void {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function startTimer(): void {
    stopTimer();
    timer = setInterval(async () => {
      if (busy || !latestCtx || !isOpenRouter(latestCtx)) return;
      await refresh(latestCtx);
    }, IDLE_REFRESH_MS);
    (timer as any).unref?.();
  }

  pi.on("session_start", async (_event, ctx) => {
    latestCtx = ctx;
    toggleFooter(ctx);
    if (readApiKey()) void refresh(ctx);
    startTimer();
  });

  pi.on("session_shutdown", async () => stopTimer());
  pi.on("before_provider_request", (event: any, ctx: any) => {
    if (!recoveryPending || !isOpenRouter(ctx)) return;
    recoveryPending = false;
    return stripOpenRouterReasoningSignatures(event.payload);
  });

  pi.on("input", (event: any) => {
    // A real user prompt starts a new recovery budget. Do not clear it for
    // the synthetic follow-up sent by this extension.
    if (event?.source !== "extension") {
      recoveryPending = false;
      recoveryAttempts = 0;
    }
  });

  pi.on("agent_start", async (_event, ctx) => { latestCtx = ctx; busy = true; });
  pi.on("agent_end", async (event: any, ctx) => {
    latestCtx = ctx;
    busy = false;
    void refresh(ctx);

    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const entries = ctx.sessionManager?.getEntries?.() || [];
    if (
      isOpenRouter(ctx)
      && recoveryAttempts < MAX_RECOVERY_ATTEMPTS
      && hasInvalidThoughtSignature(messages, entries)
    ) {
      recoveryAttempts += 1;
      recoveryPending = true;
      ctx.ui.notify(
        `OpenRouter thought signature expired; automatic recovery ${recoveryAttempts}/${MAX_RECOVERY_ATTEMPTS}...`,
        "warning",
      );
      setTimeout(() => {
        pi.sendUserMessage(
          "Continue from the last valid state. Do not repeat completed work; retry the interrupted operation.",
          { deliverAs: "followUp" },
        );
      }, 0);
    }
  });
  pi.on("model_select", async (_event, ctx) => {
    latestCtx = ctx;
    if (isOpenRouter(ctx)) {
      setTimeout(() => {
        toggleFooter(ctx);
        void refresh(ctx);
      }, 0);
    } else {
      toggleFooter(ctx);
    }
  });

  pi.registerCommand("openrouter-flex", {
    description: "Enable or disable OpenRouter Flex routing for all catalog models",
    handler: async (args, ctx) => {
      let mode = String(args || "").trim().toLowerCase();
      try {
        if (!mode) {
          const enabled = countFlexOverrides() > 0;
          const choice = await ctx.ui.select(
            `OpenRouter Flex (${enabled ? "enabled" : "disabled"})`,
            ["Enable Flex", "Disable Flex", "Cancel"],
          );
          if (choice === "Enable Flex") mode = "on";
          else if (choice === "Disable Flex") mode = "off";
          else return;
        }
        if (mode === "status") {
          ctx.ui.notify(`OpenRouter Flex overrides: ${countFlexOverrides()} (models.json)`, "info");
          return;
        }
        if (mode === "enable") mode = "on";
        if (mode === "disable") mode = "off";
        if (mode !== "on" && mode !== "off") {
          ctx.ui.notify("Usage: /openrouter-flex [on|off|status]", "warning");
          return;
        }
        const count = await setAllOpenRouterFlex(mode === "on");
        ctx.ui.notify([
          `OpenRouter Flex ${mode === "on" ? "enabled" : "disabled"} for ${count} model overrides.`,
          "Run /model or restart Pi to reload the model configuration.",
        ].join("\n"), "info");
      } catch (error: any) {
        ctx.ui.notify(`OpenRouter Flex: ${error?.message || String(error)}`, "error");
      }
    },
  });

  pi.registerCommand("openrouter", {
    description: "Show OpenRouter credit balance and usage",
    handler: async (_args, ctx) => {
      try {
        const value = await getBalance(true);
        ctx.ui.notify([
          "══ OpenRouter Balance ══",
          `remaining: ${money(value.remaining)}`,
          `total credits: ${money(value.totalCredits)}`,
          `total usage: ${money(value.totalUsage)}`,
          `(refreshed: ${new Date(value._ts).toLocaleTimeString()})`,
        ].join("\n"), "info");
      } catch (error: any) {
        ctx.ui.notify(`OpenRouter: ${error?.message || String(error)}`, "error");
      }
    },
  });

}
