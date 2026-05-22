# Simple Client

一个最小客户端，用于向服务器发送观测并打印推理速率。

你可以通过 `--env` 标志指定要使用的运行时环境。运行以下命令即可查看可用选项：

```bash
uv run examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

终端窗口 1：

```bash
uv run examples/simple_client/main.py --env DROID
```

终端窗口 2：

```bash
uv run scripts/serve_policy.py --env DROID
```
