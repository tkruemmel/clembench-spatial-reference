import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

import openai
import requests
from retry import retry

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api
from clemcore.backends.utils import ensure_messages_format, augment_response_object

logger = logging.getLogger(__name__)

NAME = "fireworks"

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MGMT_URL = "https://api.fireworks.ai/v1"

BATCH_POLL_INTERVAL = 15    # seconds between job status polls
BATCH_MAX_RETRIES = 4      # total attempts (1 original + 3 retries)
BATCH_RETRY_BASE_DELAY = 10  # seconds; doubles each attempt (10, 20, 40)


class FireworksAI(openai_api.OpenAI):
    """Backend for Fireworks AI's OpenAI-compatible API (api.fireworks.ai)."""

    def __init__(self):
        super().__init__(key_name="fireworks")

    def _make_api_client(self):
        # Non-streaming requests only send headers after generation completes;
        # a 16k-token reasoning turn can exceed the SDK's 600s default, which
        # would retry the same slow prompt forever. max_retries=0 leaves
        # retrying to the @retry decorator on generate_response.
        return openai.OpenAI(
            base_url=FIREWORKS_BASE_URL,
            api_key=self.key["api_key"],
            timeout=1800.0,
            max_retries=0,
        )

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        model_config = getattr(model_spec, "model_config", {})
        if model_config.get("batch_job"):
            account_id = self.key.get("account_id", "")
            if not account_id:
                raise ValueError(
                    "Fireworks batch_job mode requires 'account_id' in key.json under 'fireworks'."
                )
            return FireworksBatchJobModel(self.client, model_spec, account_id, self.key["api_key"])
        return FireworksModel(self.client, model_spec)


class FireworksModel(openai_api.OpenAIModel, backends.BatchGenerativeModel):
    """Serverless real-time inference via Fireworks AI."""

    def generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
            futures = {executor.submit(self.generate_response, msgs): i for i, msgs in enumerate(batch_messages)}
            results = [None] * len(batch_messages)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    @retry(
        exceptions=(openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        tries=5, delay=5, backoff=2, max_delay=60, logger=logger,
    )
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(
            model=self.model_spec.model_id,
            messages=prompt,
        )
        gen_kwargs = {**gen_kwargs, **self.gen_args}
        model_config = getattr(self.model_spec, "model_config", {})
        if "reasoning_model" in model_config:
            if not self.temperature > 0:
                raise ValueError(
                    f"For reasoning models temperature must be >0, but is {self.temperature}."
                    f"Please use the -t option to set a temperature and try again."
                )
            logger.info("Ignoring max_tokens for reasoning models, because the argument is not supported")
            del gen_kwargs["max_tokens"]

        # Session affinity: route requests sharing the same prompt prefix to the same replica
        # for higher cache hit rates on serverless deployments.
        prefix = json.dumps(prompt[0]) if prompt else ""
        gen_kwargs["user"] = hashlib.sha256(prefix.encode()).hexdigest()[:16]

        api_response = self.client.chat.completions.create(**gen_kwargs)

        usage = getattr(api_response, "usage", None)
        if usage:
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            logger.info(
                "Fireworks usage — prompt: %d (cached: %d), completion: %d",
                usage.prompt_tokens or 0, cached, usage.completion_tokens or 0,
            )

        message = api_response.choices[0].message
        if message.role != "assistant":
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")

        response_text = (message.content or "").strip()
        if not response_text:
            logger.warning("OpenAI API response message content is None or empty, returning empty string.")

        response = api_response.model_dump(mode="json")
        return prompt, response, response_text


class FireworksBatchJobModel(FireworksModel, backends.BatchGenerativeModel):
    """Async batch inference jobs via Fireworks Batch API (~50% cost reduction).

    Requires 'account_id' in key.json under 'fireworks'. Enable by setting
    "model_config": {"batch_job": true} in the model registry entry.
    """

    def __init__(self, client, model_spec: backends.ModelSpec, account_id: str, api_key: str):
        super().__init__(client, model_spec)
        self.account_id = account_id
        self.api_key = api_key

    def _auth_headers(self) -> Dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _json_headers(self) -> Dict:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    def _mgmt_url(self, path: str) -> str:
        return f"{FIREWORKS_MGMT_URL}/accounts/{self.account_id}/{path}"

    def generate_batch_response(self, batch_messages: List[List[Dict]]) -> List[Tuple[Any, Any, str]]:
        # Encode once — no side effects, no reason to repeat on retry.
        encoded_prompts = []
        jsonl_lines = []
        model_config = getattr(self.model_spec, "model_config", {})
        for i, messages in enumerate(batch_messages):
            prompt = self.encode_messages(messages)
            encoded_prompts.append(prompt)
            body = {"messages": prompt, **self.gen_args}
            if "reasoning_model" in model_config:
                body.pop("max_tokens", None)
            jsonl_lines.append(json.dumps({"custom_id": str(i), "body": body}))
        jsonl_bytes = "\n".join(jsonl_lines).encode()

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(BATCH_MAX_RETRIES):
            try:
                return self._run_batch_job(encoded_prompts, jsonl_bytes)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status >= 500 and attempt < BATCH_MAX_RETRIES - 1:
                    delay = BATCH_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Batch job HTTP %d on attempt %d/%d — retrying in %ds.",
                        status, attempt + 1, BATCH_MAX_RETRIES, delay,
                    )
                    last_exc = exc
                    time.sleep(delay)
                else:
                    raise
            except requests.exceptions.ConnectionError as exc:
                if attempt < BATCH_MAX_RETRIES - 1:
                    delay = BATCH_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Batch job connection error on attempt %d/%d — retrying in %ds: %s",
                        attempt + 1, BATCH_MAX_RETRIES, delay, exc,
                    )
                    last_exc = exc
                    time.sleep(delay)
                else:
                    raise
        raise last_exc

    def _run_batch_job(
        self, encoded_prompts: List[Any], jsonl_bytes: bytes
    ) -> List[Tuple[Any, Any, str]]:
        """One attempt: create dataset, upload, submit job, poll, download, parse."""
        batch_id = uuid.uuid4().hex[:8]
        input_dataset_id = f"clem-input-{batch_id}"
        output_dataset_id = f"clem-output-{batch_id}"
        job_id = f"clem-job-{batch_id}"

        # Create input dataset
        resp = requests.post(
            self._mgmt_url("datasets"),
            headers=self._json_headers(),
            json={"datasetId": input_dataset_id, "dataset": {"userUploaded": {}, "exampleCount": len(encoded_prompts)}},
        )
        if not resp.ok:
            raise RuntimeError(f"Dataset creation failed {resp.status_code}: {resp.text}")
        logger.info("Created input dataset %s", input_dataset_id)

        # Upload JSONL
        resp = requests.post(
            self._mgmt_url(f"datasets/{input_dataset_id}:upload"),
            headers=self._auth_headers(),
            files={"file": ("batch.jsonl", jsonl_bytes, "application/jsonl")},
        )
        if not resp.ok:
            raise RuntimeError(f"Upload failed {resp.status_code}: {resp.text}")
        logger.info("Uploaded %d requests to dataset %s", len(encoded_prompts), input_dataset_id)

        # Submit batch job — this is the call that 500'd in production
        resp = requests.post(
            self._mgmt_url(f"batchInferenceJobs?batchInferenceJobId={job_id}"),
            headers=self._json_headers(),
            json={
                "model": self.model_spec.model_id,
                "inputDatasetId": f"accounts/{self.account_id}/datasets/{input_dataset_id}",
                "outputDatasetId": f"accounts/{self.account_id}/datasets/{output_dataset_id}",
                "inferenceParameters": {},
            },
        )
        resp.raise_for_status()
        logger.info("Submitted batch job %s", job_id)

        # Poll until complete
        while True:
            resp = requests.get(self._mgmt_url(f"batchInferenceJobs/{job_id}"), headers=self._auth_headers())
            resp.raise_for_status()
            job = resp.json()
            state = job.get("state", "UNKNOWN")
            logger.info("Batch job %s state: %s", job_id, state)
            if state in ("COMPLETED", "JOB_STATE_COMPLETED"):
                break
            if state in ("FAILED", "EXPIRED", "JOB_STATE_FAILED", "JOB_STATE_EXPIRED"):
                status = job.get("status") or {}
                detail = status.get("message") or status.get("code") or ""
                suffix = f" ({detail})" if detail else ""
                raise RuntimeError(f"Fireworks batch job {job_id} ended with state: {state}{suffix}")
            time.sleep(BATCH_POLL_INTERVAL)

        # Download results
        resp = requests.get(
            self._mgmt_url(f"datasets/{output_dataset_id}:getDownloadEndpoint"),
            headers=self._json_headers(),
            json={},
        )
        resp.raise_for_status()
        signed_urls = resp.json().get("filenameToSignedUrls", {})

        results_by_id: Dict[str, Any] = {}
        logged_sample = False
        for filename, url in signed_urls.items():
            if "error" in filename.lower():
                continue
            content = requests.get(url).text
            for line in content.strip().splitlines():
                if line:
                    item = json.loads(line)
                    if not logged_sample:
                        logger.debug("Batch output sample keys: %s", list(item.keys()))
                        logger.debug("Batch output response keys: %s", list(item.get("response", {}).keys()) if isinstance(item.get("response"), dict) else type(item.get("response")))
                        logged_sample = True
                    results_by_id[item["custom_id"]] = item

        # Assemble return tuples in original order
        results = []
        for i, prompt in enumerate(encoded_prompts):
            item = results_by_id.get(str(i))
            if item is None:
                raise RuntimeError(f"Fireworks batch job {job_id}: no result for request {i}")
            if item.get("error"):
                raise RuntimeError(f"Fireworks batch job {job_id}: request {i} failed: {item['error']}")

            response = item["response"]
            response_body = response.get("body", response) if isinstance(response, dict) else response
            message = response_body["choices"][0]["message"]
            if message["role"] != "assistant":
                raise AttributeError(f"Response role is {message['role']!r}, expected 'assistant'")

            response_text = (message.get("content") or "").strip()
            if not response_text:
                logger.warning("Fireworks batch response for request %d is empty.", i)

            results.append((prompt, response_body, response_text))

        return results
