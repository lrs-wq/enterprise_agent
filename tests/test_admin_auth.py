import pytest
from fastapi import HTTPException
from starlette.requests import Request

from company_agent.config import Settings
from company_agent.main import _client_key, admin_auth_limiter, require_admin


def make_request(host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/admin/documents",
            "headers": [],
            "client": (host, 12345),
        }
    )


async def test_admin_api_is_locked_when_token_is_not_configured() -> None:
    request = make_request("192.0.2.10")
    with pytest.raises(HTTPException) as raised:
        await require_admin(request, Settings(admin_token=""), None)

    assert raised.value.status_code == 503


async def test_admin_api_requires_matching_token() -> None:
    token = "a-secure-local-token-that-is-long"
    settings = Settings(admin_token=token)
    request = make_request("192.0.2.11")

    with pytest.raises(HTTPException) as raised:
        await require_admin(request, settings, "wrong")

    assert raised.value.status_code == 401
    assert await require_admin(request, settings, token) is None


async def test_admin_auth_failures_are_rate_limited() -> None:
    token = "a-secure-local-token-that-is-long"
    settings = Settings(admin_token=token)
    request = make_request("192.0.2.12")
    key = _client_key(request)
    admin_auth_limiter.reset(key)
    try:
        statuses = []
        for _ in range(settings.admin_auth_failure_limit):
            with pytest.raises(HTTPException) as raised:
                await require_admin(request, settings, "wrong")
            statuses.append(raised.value.status_code)
        assert statuses[:-1] == [401] * (settings.admin_auth_failure_limit - 1)
        assert statuses[-1] == 429
        assert raised.value.headers is not None
        assert int(raised.value.headers["Retry-After"]) >= 1
    finally:
        admin_auth_limiter.reset(key)


async def test_previous_admin_token_only_works_during_rotation_window() -> None:
    current = "current-secure-local-token-value"
    previous = "previous-secure-local-token-value"
    request = make_request("192.0.2.13")
    active_rotation = Settings(
        admin_token=current,
        admin_previous_token=previous,
        admin_previous_token_expires_at="2099-01-01T00:00:00Z",
    )
    expired_rotation = Settings(
        admin_token=current,
        admin_previous_token=previous,
        admin_previous_token_expires_at="2020-01-01T00:00:00Z",
    )

    assert await require_admin(request, active_rotation, previous) is None
    with pytest.raises(HTTPException) as raised:
        await require_admin(request, expired_rotation, previous)
    assert raised.value.status_code == 401
    admin_auth_limiter.reset(_client_key(request))
