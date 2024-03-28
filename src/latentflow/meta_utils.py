import json

def write_meta(meta, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps(meta))


def read_meta(file_path):
    meta = {}
    with open(file_path, 'r') as f:
        content = f.read()
        if content != '':
            candidate = json.loads(content)
            if isinstance(candidate, dict) or isinstance(candidate, list):
                meta = candidate
    return meta
