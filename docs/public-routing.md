# 公网路由运行说明

## 访问地址

- 公网聊天页：`https://ai.hzzootech.com/company-ai/`
- 公网管理页：`https://ai.hzzootech.com/company-ai/admin`
- 本机聊天页：`http://127.0.0.1:8000/`

服务复用已备案域名的独立 `/company-ai/` 路径，不覆盖服务器上的既有应用。

## 路由结构

公网 HTTPS 请求先到服务器 Nginx。Nginx 转发到服务器回环端口
`127.0.0.1:18000`，该端口由本机发起的反向 SSH 隧道连接至
`127.0.0.1:8000`。模型、知识库和上传文件仍保留在本机。

## 本机常驻服务

macOS LaunchAgent 负责两个进程：

- `com.gogelila.company-ai-agent`：启动本地 FastAPI 服务。
- `com.gogelila.company-ai-agent-tunnel`：建立并保持反向 SSH 隧道。

由于 macOS 会限制后台进程读取“桌面”目录，公网运行副本位于
`~/Library/Application Support/GogelilaAI/company-ai-agent`。代码更新后运行
`bash scripts/sync-public-runtime.sh`，再重启应用 LaunchAgent。运行副本中的
`data/` 是公网服务的实时知识库数据目录。

状态检查：

```bash
launchctl print gui/$(id -u)/com.gogelila.company-ai-agent
launchctl print gui/$(id -u)/com.gogelila.company-ai-agent-tunnel
```

日志位于运行副本的 `data/company-ai-agent.*.log` 和
`data/public-tunnel.*.log`。

## 安全边界

- 公网只开放 Nginx 的 80/443；SSH 隧道端口仅监听服务器回环地址。
- 管理 API 仍要求 `AI_AGENT_ADMIN_TOKEN`。
- Nginx 终止 TLS，并将 HTTP 自动跳转至 HTTPS。
- 精确聊天路径 `/company-ai/api/chat` 和 `/company-ai/api/chat/stream` 限制为
  64 KiB 请求体、85 秒读取超时；应用在 75 秒停止排队、检索、生成和修正，代理
  预留 10 秒交付最终结果。SSE 路径还须关闭代理缓冲，才能及时显示真实进度。
- 其他管理路径保留 100 MiB 请求体和 300 秒代理超时，对齐应用 280 秒后台任务预算。
- 应用内部还实施单路聊天/模型并发、有界队列、公开请求限速、管理员失败限速和上传解析上限；
  代理限制不是唯一保护层。
- 本机或公网服务器离线时，公网服务暂时不可用；LaunchAgent 会自动重连。

当前服务器配置备份为
`/etc/nginx/snippets/company-ai-location.conf.pre-o04-20260921T074510Z`。修改配置后必须先运行
`nginx -t`，成功后才 reload。

2026-09-28 开发目录已为 SSE 增加精确路径配置，但尚未同步独立公网副本；上述 SSE
规则需部署并经公网首事件/取消冒烟后才可视为线上生效。
