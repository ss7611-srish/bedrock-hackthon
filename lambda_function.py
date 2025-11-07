import os, json, boto3, logging, traceback
from botocore.exceptions import ClientError
from datetime import datetime
from decimal import Decimal

# ---------- Logging ----------
log = logging.getLogger()
log.setLevel(logging.INFO)

# ---------- Config from env ----------
REGION      = (os.environ.get("AWS_REGION")
               or boto3.session.Session().region_name
               or "us-east-1")
KB_ID       = os.environ.get("KB_ID")                 # e.g., FDK2ED8JT2
MODEL_ID    = os.environ.get("MODEL_ID")              # e.g., meta.llama3-8b-instruct-v1:0
TABLE       = os.environ.get("TABLE", "hiera")
FILTER_KEY  = os.environ.get("FILTER_KEY", "tab")     # e.g., x-amz-meta-tab OR tab

# ---------- AWS clients ----------
bedrock_rt = boto3.client("bedrock-runtime",       region_name=REGION)
agent_rt   = boto3.client("bedrock-agent-runtime", region_name=REGION)
dynamo     = boto3.resource("dynamodb",            region_name=REGION).Table(TABLE)

# ---------- Helpers ----------
def to_dynamo(value):
    """Recursively convert floats to Decimal so DynamoDB accepts them."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [to_dynamo(v) for v in value]
    if isinstance(value, dict):
        return {k: to_dynamo(v) for k, v in value.items()}
    return value

def _kb_retrieve(query: str, tab: str):
    """
    Retrieve chunks from the Knowledge Base filtered by metadata FILTER_KEY=tab.
    If zero results, logs unfiltered hits (to reveal actual metadata keys).
    Returns (ctx_text, citations).
    """
    def _call(filter_obj=None):
        conf = {
            "knowledgeBaseId": KB_ID,
            "retrievalQuery": {"text": query},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 10
                }
            }
        }
        if filter_obj:
            conf["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = filter_obj
        return agent_rt.retrieve(**conf)

    # 1) Try with metadata filter
    filt = {"equals": {"key": FILTER_KEY, "value": tab}}
    resp = _call(filt)
    results = resp.get("retrievalResults", [])
    log.info(f"KB filtered FILTER_KEY={FILTER_KEY!r} tab={tab!r} count={len(results)}")

    # 2) If none, run unfiltered and print attrs so we can see the real keys
    if not results:
        resp_all = _call()
        all_results = resp_all.get("retrievalResults", [])
        log.info(f"KB unfiltered count={len(all_results)}")
        for i, r in enumerate(all_results[:10]):
            uri = (r.get("location", {}) or {}).get("s3Location", {}).get("uri")
            md = r.get("metadata") or {}
            # Custom metadata often appears under 'attributes' or 'documentMetadata'
            attrs = md.get("attributes") or md.get("documentMetadata") or md
            log.info(f"r{i} uri={uri} attrs={attrs}")

        # Optional folder-scope fallback (helps while wiring up metadata):
        scoped = []
        for r in all_results:
            uri = (r.get("location", {}) or {}).get("s3Location", {}).get("uri", "")
            if f"/tabs/{tab}/" in uri:
                scoped.append(r)
        if scoped:
            log.info(f"Using folder-scope fallback results={len(scoped)}")
            results = scoped

    # Build a concatenated context block + light citations (location + score)
    chunks, cites = [], []
    for r in results:
        content = r.get("content", {})
        text = ""
        if isinstance(content, dict):
            text = content.get("text", "")
        elif isinstance(content, list):
            text = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if text:
            chunks.append(text)
        cites.append({
            "location": r.get("location", {}),
            "score": r.get("score")
        })

    ctx_text = "\n\n---\n\n".join(chunks) if chunks else ""
    return ctx_text, cites

def _converse(system_text: str, user_text: str) -> str:
    """Call Bedrock /converse with a single user message (system folded in)."""
    if not MODEL_ID:
        raise RuntimeError("MODEL_ID env var is required (e.g., meta.llama3-8b-instruct-v1:0)")
    merged = f"Instructions: {system_text}\n\nUser: {user_text}"
    out = bedrock_rt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": merged}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2}
    )
    return out["output"]["message"]["content"][0]["text"]

def _put_message(user_id: str, tab: str, role: str, text: str, answer=None, citations=None):
    item = {
        "pk": f"USER#{user_id}#TAB#{tab}",
        "sk": f"MSG#{datetime.utcnow().isoformat()}",
        "type": "message",
        "role": role,
        "text": text,
    }
    if answer is not None:
        item["answer"] = answer
    if citations is not None:
        item["citations"] = citations
    dynamo.put_item(Item=to_dynamo(item))

# ---------- Handler ----------
def lambda_handler(event, context):
    try:
        log.info(f"REGION={REGION} KB_ID={KB_ID} MODEL_ID={MODEL_ID} FILTER_KEY={FILTER_KEY}")

        # Accept API Gateway proxy (event["body"] is a JSON string) or direct dict
        body = event.get("body") if isinstance(event, dict) else event
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = {}
        if not isinstance(body, dict):
            body = {}

        tab = (body.get("tab") or "courses").lower()
        user_id = body.get("userId", "demo")
        message = (body.get("message") or "What should I know now?").strip()

        if tab not in {"courses", "interview", "personal"}:
            return {"statusCode": 400,
                    "body": json.dumps({"error": "tab must be one of: courses | interview | personal"})}

        # Log user message
        _put_message(user_id, tab, "user", message)

        # Retrieve context (KB)
        ctx, cites = _kb_retrieve(message, tab)

        # Generate answer
        system = f"You are the '{tab}' assistant. Use the Context when helpful. Be concise and actionable."
        answer = _converse(system, f"{message}\n\nContext:\n{ctx}")

        # Log assistant message
        _put_message(user_id, tab, "assistant", message, answer, cites)

        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": "*"},
            "body": json.dumps({"tab": tab, "answer": answer, "sources": cites})
        }

    except ClientError as e:
        log.error(f"AWS ClientError: {e.response}")
        log.error(traceback.format_exc())
        return {"statusCode": 500,
                "headers": {"content-type": "application/json",
                            "access-control-allow-origin": "*"},
                "body": json.dumps({"error": str(e)})}
    except Exception as e:
        log.error(f"Unhandled: {e}")
        log.error(traceback.format_exc())
        return {"statusCode": 500,
                "headers": {"content-type": "application/json",
                            "access-control-allow-origin": "*"},
                "body": json.dumps({"error": str(e)})}
