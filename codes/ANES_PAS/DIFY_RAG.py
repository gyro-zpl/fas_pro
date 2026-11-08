import dashscope
from http import HTTPStatus
import os
import requests
import json
import csv
import argparse
# Read API keys from environment variables to avoid hardcoding
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


def dify_stream_from_curl(prompt):
    """
    Use Python to reproduce a curl request to Dify /api/v1/chat-messages, printing streaming responses.
    Environment variables required:
      - DIFY_API_URL, e.g., http://10.113.178.150:2001/api/v1/chat-messages
      - DIFY_API_KEY, e.g., app-xxxxx
    """
    api_url = os.environ.get("DIFY_API_URL", "")
    api_key = os.environ.get("DIFY_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json;q=0.9"
    }

    payload = {
        "inputs": {},
        "query": prompt,
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "gyro"
    }

    with requests.post(api_url, headers=headers, json=payload, stream=True, timeout=60) as r:
        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            print("ERROR: Received an HTML page. Check if you hit the console instead of /api. status=", r.status_code)
            print(r.text[:500])
            return

        r.raise_for_status()

        # SSE streaming
        if "text/event-stream" in content_type:
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                line = chunk.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                try:
                    obj = json.loads(line)
                    # Print only the 'answer' field in real-time
                    event_type = obj.get("event")
                    if "answer" in obj:
                        print(obj["answer"], end="", flush=True)
                    # Print a newline when the message ends
                    if event_type == "message_end":
                        print("")
                except json.JSONDecodeError:
                    # Ignore non-JSON chunks
                    pass
        else:
            # Non-SSE case: print full JSON
            try:
                obj = r.json()
                if isinstance(obj, dict) and "answer" in obj:
                    print(obj["answer"])
                else:
                    print(json.dumps(obj, ensure_ascii=False))
            except ValueError:
                print(r.text[:2000])

def clean_model_output(text: str):
    """Clean model output and try to extract a JSON array (rows with 3 items '1/2/3') or an object with a 'rows' field."""
    s = (text or "").strip()
    # Remove markdown fences if present
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    # 1) Direct parse
    try:
        data = json.loads(s)
        return data
    except Exception:
        pass

    # 2) Extract an array like [[...],[...]]
    import re
    array_pattern = r"\[\s*\[.*?\]\s*\]"
    matches = re.findall(array_pattern, s, re.DOTALL)
    if matches:
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue

    # 3) Extract rows with exactly three elements
    row_pattern = r"\[\s*[\"\']?(\d+)[\"\']?\s*,\s*[\"\']?(\d+)[\"\']?\s*,\s*[\"\']?(\d+)[\"\']?\s*\]"
    rows = re.findall(row_pattern, s)
    if rows:
        return [[r[0], r[1], r[2]] for r in rows]

    raise ValueError("Failed to extract valid JSON data from model output")

def extract_rows(obj):
    """Extract 2D array rows from object. Supports {'rows': [...]} or a raw 2D array."""
    if isinstance(obj, dict) and "rows" in obj:
        return obj["rows"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Returned JSON is not an array and does not contain a 'rows' field")

def validate_rows(rows):
    """Validate each row has length 3 and elements are only '1'/'2'/'3'. Return normalized 2D array (strings)."""
    valid = {"1", "2", "3"}
    cleaned = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Each row must contain exactly 3 items")
        row_str = list(map(lambda x: str(x).strip(), row))
        if not all(item in valid for item in row_str):
            raise ValueError("Elements must be one of '1'/'2'/'3'")
        cleaned.append(row_str)
    return cleaned

def save_rows_to_csv(rows, filename):
    """Write CSV with headers natenvir, natheal, natcity; write numbers only, cleaned with lambda."""
    headers_str = os.environ.get("COLUMNS", '')
    headers = json.loads(headers_str)
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            clean_row = list(map(lambda x: int(x) if str(x).isdigit() else x, row))
            w.writerow(clean_row)

def dify_request_non_stream(prompt: str, conversation_id = None):
    """Non-stream request to Dify. Return the 'answer' string if present, otherwise the full JSON response."""
    api_url = os.environ.get("DIFY_API_URL", "")
    api_key = os.environ.get("DIFY_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "conversation_id": conversation_id,
        "user": "gyro"
    }

    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    try:
        obj = r.json()
    except ValueError:
        # Not JSON
        return r.text

    #print(obj['conversation_id'])
    # Typical Dify response: {"answer": "..."} or more complex structures
    print(obj)
    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"], obj['conversation_id']

    
    return obj, None

def run_once(prompt: str, conversation_id = None):
    """Call non-stream API, parse and validate rows, return 2D array (rows)."""
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id)
    #print(raw)
    
    return raw, conversation_id

def run_once_to_csv(prompt: str, out_csv: str):
    """Backward compatibility: single run and save to CSV."""
    rows,_,_ = run_once(prompt)
    save_rows_to_csv(rows, out_csv)
    return out_csv

def save_raw_to_csv(raw_rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(raw_rows)

def main(mission, batch_size, epochs, out_csv="gss_all_epochs.csv", out_raw_path="gss_all_epochs_raw.txt"):
    all_rows = []
    raw_rows = []
    conversation_id = None
    for epoch in range(epochs):
        prompt = f"Mission: {mission}. Now generate batch No.{epoch}. Sample batch size is {batch_size}."
        try:
            rows,raws,conversation_id = run_once(prompt, conversation_id)
            all_rows.extend(rows)
            raw_rows.extend(raws)
            print("="*100)
            print(f"Epoch {epoch}: +{len(rows)} rows (total {len(all_rows)})")
            
            print(f"Epoch {epoch}: +{len(raws)} raw rows (total {len(raw_rows)})")
        except Exception as e:
            print(f"ERROR @ epoch {epoch}: {e}")
    if all_rows:
        save_rows_to_csv(all_rows, out_csv)
        print(f"Saved all {len(all_rows)} rows to {out_csv}")
    # if raw_rows:
    #     save_raw_to_csv(raw_rows, out_raw_path)
    #     print(f"Saved all {len(raw_rows)} raw rows to {out_raw_path}")

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=str, default="edc")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out_csv", type=str, default="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_edc.csv")
    parser.add_argument("--out_raw_path", type=str, default=None)
    return parser.parse_args()
if __name__ == '__main__':
    # Aggregate all epochs into one CSV
    #(batchsize, epochs, out_csv, out_raw_path)
    parser = parser_args()
    main(parser.mission, parser.batch_size, parser.epochs, parser.out_csv, parser.out_raw_path)
