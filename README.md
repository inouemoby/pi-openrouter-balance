# pi-openrouter-balance

OpenRouter credits and balance monitor for [Pi](https://pi.dev).

Shows the current OpenRouter balance beside Pi's token and context statistics when an OpenRouter model is active. Also provides `/openrouter` and the `openrouter_balance` tool.

## Install

```bash
pi install git:github.com/inouemoby/pi-openrouter-balance
```

## Authentication

Use an OpenRouter API key already configured in Pi:

```text
/login openrouter
```

Or set:

```text
OPENROUTER_API_KEY
```

The extension reads the key from Pi's standard `auth.json` provider entry or the environment variable. It does not display or send the key anywhere except OpenRouter's official credits endpoint.

## Features

- Footer balance display, only when the active provider is `openrouter`;
- `/openrouter` detailed balance and usage command;
- `/openrouter-flex` opens an enable/disable submenu; `/openrouter-flex on|off|status` executes directly without opening the submenu; it dynamically applies or removes `service_tier: "flex"` for all current OpenRouter catalog models;
- `openrouter_balance` tool for agent-accessible balance checks;
- automatic refresh after each agent turn and every five minutes while idle;
- standard root `index.ts` extension entry, with no build directory required.

## Footer preview

```text
~/project (main) • session
↑14k ↓2.1k 12.5%/1.0M (auto) $0.0112 OR:$4.9948   (openrouter) google/gemini-3.8-flash
```

`OR:` is the remaining OpenRouter credit balance. It is not the Google AI Pro or Antigravity quota.

## Data source

The extension reads `GET https://openrouter.ai/api/v1/credits` and uses:

The Flex command reads the current OpenRouter model catalog and updates Pi's standard `~/.pi/agent/models.json` `modelOverrides`; it does not create a second provider or model catalog.

```text
remaining = total_credits - total_usage
```

## License

MIT
