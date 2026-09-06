"""
Mock-based test script for app/services/pixazo_service.py.

No real network calls are made: a fake HTTP client double (exposing only
.post()/.get(), matching the subset of httpx.Client's interface this
service uses) is injected directly into PixazoImageService.

Run from backend/:
    python ../tests/test_pixazo_service_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.pixazo_service import (  # noqa: E402
    PixazoAPIError,
    PixazoConfigurationError,
    PixazoImageService,
    PixazoTimeoutError,
)

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def dummy_settings() -> Settings:
    return Settings(_env_file=None, PIXAZO_API_KEY="fake-pixazo-key", PIXAZO_IMAGE_MODEL="z-image-base")


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeHttpClient:
    """
    Simulates the submit -> poll(s) -> COMPLETED flow (or a configured
    failure), tracking calls for assertions.
    """

    def __init__(self, poll_statuses=None, submit_error=False):
        self.poll_statuses = poll_statuses or ["COMPLETED"]
        self.submit_error = submit_error
        self.post_calls = []
        self.get_calls = []
        self._poll_index = 0

    def post(self, url, headers=None, json=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json})
        if self.submit_error:
            return FakeResponse(400, {"error": "Model not found", "message": "Model 'x' not found"})
        return FakeResponse(
            200,
            {"request_id": "req-123", "status": "QUEUED", "polling_url": "https://gateway.pixazo.ai/v2/requests/status/req-123"},
        )

    def get(self, url, headers=None):
        self.get_calls.append({"url": url, "headers": headers})
        status = self.poll_statuses[min(self._poll_index, len(self.poll_statuses) - 1)]
        self._poll_index += 1

        if status == "COMPLETED":
            return FakeResponse(
                200,
                {
                    "request_id": "req-123",
                    "status": "COMPLETED",
                    "output": {"media_url": ["https://example.com/generated.png"]},
                },
            )
        if status in ("FAILED", "ERROR"):
            return FakeResponse(200, {"request_id": "req-123", "status": status, "error": "generation failed"})
        return FakeResponse(200, {"request_id": "req-123", "status": status})


def test_1_successful_generation():
    print("\n=== Test 1: successful submit -> poll -> COMPLETED returns the image URL ===")
    client = FakeHttpClient(poll_statuses=["QUEUED", "PROCESSING", "COMPLETED"])
    service = PixazoImageService(settings=dummy_settings(), client=client)

    result = service.generate_image("a lighthouse at dawn, watercolor")

    check(result.image_url == "https://example.com/generated.png", "the completed job's media_url is returned")
    check(result.model == "z-image-base", "the result records which model was used")
    check(len(client.post_calls) == 1, "exactly one submit request was made")
    check(
        client.post_calls[0]["url"] == "https://gateway.pixazo.ai/z-image-base/v1/z-image-base-request",
        "the submit URL follows the documented {model}/v1/{model}-request pattern",
    )
    check(
        client.post_calls[0]["headers"]["Ocp-Apim-Subscription-Key"] == "fake-pixazo-key",
        "the API key is sent via the documented Ocp-Apim-Subscription-Key header",
    )
    check(client.post_calls[0]["json"] == {"prompt": "a lighthouse at dawn, watercolor"}, "the prompt was sent in the submit body")
    check(len(client.get_calls) == 3, "polling continued until COMPLETED (3 polls: QUEUED, PROCESSING, COMPLETED)")


def test_2_missing_api_key():
    print("\n=== Test 2: missing API key is handled clearly, no HTTP call made ===")
    empty_settings = Settings(_env_file=None, PIXAZO_API_KEY=None)
    client = FakeHttpClient()
    service = PixazoImageService(settings=empty_settings, client=client)

    raised = False
    try:
        service.generate_image("anything")
    except PixazoConfigurationError:
        raised = True

    check(raised, "PixazoConfigurationError was raised when no API key is configured")
    check(client.post_calls == [], "no submit request was made without a configured key")


def test_3_job_failure_is_surfaced():
    print("\n=== Test 3: a FAILED/ERROR job status raises PixazoAPIError with the reason ===")
    client = FakeHttpClient(poll_statuses=["FAILED"])
    service = PixazoImageService(settings=dummy_settings(), client=client)

    raised = False
    message = ""
    try:
        service.generate_image("anything")
    except PixazoAPIError as exc:
        raised = True
        message = str(exc)

    check(raised, "PixazoAPIError was raised for a FAILED job")
    check("generation failed" in message, "the underlying failure reason is included in the error")


def test_4_submit_error_is_surfaced():
    print("\n=== Test 4: a non-2xx submit response raises PixazoAPIError ===")
    client = FakeHttpClient(submit_error=True)
    service = PixazoImageService(settings=dummy_settings(), client=client)

    raised = False
    try:
        service.generate_image("anything")
    except PixazoAPIError:
        raised = True

    check(raised, "PixazoAPIError was raised for a non-2xx submit response")


def test_5_poll_timeout():
    print("\n=== Test 5: a job stuck QUEUED forever raises PixazoTimeoutError ===")
    client = FakeHttpClient(poll_statuses=["QUEUED"])  # never completes
    service = PixazoImageService(settings=dummy_settings(), client=client)

    # Patch the poll interval/attempts down so this test doesn't actually sleep ~60s.
    import app.services.pixazo_service as pixazo_module
    original_interval = pixazo_module._POLL_INTERVAL_SECONDS
    original_attempts = pixazo_module._MAX_POLL_ATTEMPTS
    pixazo_module._POLL_INTERVAL_SECONDS = 0.01
    pixazo_module._MAX_POLL_ATTEMPTS = 3
    try:
        raised = False
        try:
            service.generate_image("anything")
        except PixazoTimeoutError:
            raised = True
        check(raised, "PixazoTimeoutError was raised after exhausting the poll budget")
        check(len(client.get_calls) == 3, "polling stopped exactly at the configured max attempts")
    finally:
        pixazo_module._POLL_INTERVAL_SECONDS = original_interval
        pixazo_module._MAX_POLL_ATTEMPTS = original_attempts


if __name__ == "__main__":
    test_1_successful_generation()
    test_2_missing_api_key()
    test_3_job_failure_is_surfaced()
    test_4_submit_error_is_surfaced()
    test_5_poll_timeout()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All Pixazo service tests passed.")
