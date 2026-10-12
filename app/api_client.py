"""Small, timeout-bounded client for the existing FastAPI service."""

from __future__ import annotations

from typing import Any

import requests


class APIClientError(RuntimeError):
    """Safe, user-facing API failure message."""


def get_health(base_url: str, *, timeout: float = 3.0) -> dict[str, Any] | None:
    """Return service readiness without exposing request or host exceptions."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        if not response.ok:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except requests.RequestException:
        return None
    except ValueError:
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
    timeout: float = 15.0,
) -> dict[str, Any]:
    """POST a borrower payload and return JSON with safe error translation."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise APIClientError("The risk service took too long to respond. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise APIClientError(
            "The risk service is unavailable. Check that the API is running and try again."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("Could not contact the risk service. Please try again.") from exc

    if response.status_code == 422:
        raise APIClientError(_validation_message(response))
    if response.status_code in (503, 502, 504):
        raise APIClientError("The requested service is not ready. Check the API health status.")
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
