#!/usr/bin/env python3
"""Smoke-test a running local Company AI Agent instance."""

import argparse
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查本地 AI Agent 服务")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="请介绍公司及主要产品。")
    parser.add_argument("--skip-chat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=90) as client:
            health = client.get(f"{base_url}/api/health")
            health.raise_for_status()
            health_data = health.json()
            print(f"健康检查：{health_data}")
            if health_data.get("status") != "ok":
                print("服务处于 degraded 状态。", file=sys.stderr)
                return 1
            if args.skip_chat:
                return 0
            response = client.post(
                f"{base_url}/api/chat",
                json={"question": args.question, "history": []},
            )
            response.raise_for_status()
            payload = response.json()
            print(f"回答状态：{payload['status']}")
            print(f"回答标题：{payload['answer']['title']}")
            print(f"引用数量：{len(payload['sources'])}")
            print(f"耗时：{payload['metrics']}")
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"冒烟测试失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
