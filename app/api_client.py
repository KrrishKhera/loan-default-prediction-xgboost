"""Small, timeout-bounded client for the existing FastAPI service."""

from __future__ import annotations

import time
from typing import Any

import requests


class APIClientError(RuntimeError):
    """Safe, user-facing API failure message."""


def get_health(
    base_url: str, *, timeout: float | tuple[float, float] = (1.5, 2.0)
) -> dict[str, Any] | None:
    """Return health only when the existing prediction readiness contract passes."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        if not response.ok:
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("status") != "ok"
            or payload.get("model_loaded") is not True
            or not isinstance(payload.get("explanation_ready"), bool)
        ):
            return None
        return payload
    except requests.RequestException:
        return None
    except ValueError:
        return None


def get_health_with_retry(
    base_url: str,
    *,
    attempts: int = 3,
    timeout: float | tuple[float, float] = (1.0, 2.0),
    delay: float = 0.4,
) -> dict[str, Any] | None:
    """Retry readiness a bounded number of times, pausing briefly between checks."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        result = get_health(base_url, timeout=timeout)
        if result is not None:
            return result
        if attempt + 1 < attempts and delay > 0:
            time.sleep(delay)
    return None


def _validation_message(response: requests.Response) -> str:
    try:
        details = response.json().get("detail", [])
    except (ValueError, AttributeError):
        return "Some values were rejected. Check the form and try again."
    if isinstance(details, list):
        fields = [
            ".".join(str(part) for part in item.get("loc", [])[1:])
            for item in details
            if isinstance(item, dict) and item.get("loc")
        ]
        fields = [field for field in fields if field]
        if fields:
            return "Please check these fields: " + ", ".join(dict.fromkeys(fields))
    return "Some values were rejected. Check the form and try again."


def post_prediction(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float | tuple[float, float] = (2.0, 15.0),
) -> dict[str, Any]:
    """POST a borrower payload and return JSON with safe error translation."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise APIClientError(
            "The prediction service is still starting or responding slowly. Please wait a moment and retry."
        ) from exc
    except requests.ConnectionError as exc:
        raise APIClientError(
            "The prediction service is not responding yet. Please wait a moment and retry."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("Could not contact the risk service. Please try again.") from exc

    if response.status_code == 422:
        raise APIClientError(_validation_message(response))
    if response.status_code in (503, 502, 504):
        raise APIClientError(
            "The prediction service is still starting. Please wait a moment and retry."
        )
    if response.status_code >= 500:
        raise APIClientError("The risk service encountered an error. Please try again later.")
    if not response.ok:
        raise APIClientError("The risk service rejected the request. Check the form and try again.")
    try:
        result = response.json()
    except ValueError as exc:
        raise APIClientError("The risk service returned an unreadable response.") from exc
    if not isinstance(result, dict):
        raise APIClientError("The risk service returned an unexpected response format.")
    return result
