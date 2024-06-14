import json
from omegaconf import OmegaConf

def write_meta(meta, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps(meta))


def read_meta(file_path, context=None):
    meta = {}
    if '.json' in file_path:
        with open(file_path, 'r') as f:
            content = f.read()
            if content != '':
                candidate = json.loads(content)
                if isinstance(candidate, dict) or isinstance(candidate, list):
                    meta = candidate
    elif '.yml' in file_path or '.yaml' in file_path:
        meta = OmegaConf.to_container(OmegaConf.load(file_path), resolve=True)

    if context is not None:
        c = {}
        c.update(context)
        for k in meta:
            if isinstance(meta[k], str):
                c[k] = meta[k]
        print('context', c)
        meta = change_placeholders(meta, c)

    return meta

def change_placeholders(meta, context):
    if isinstance(meta, dict):
        newkeys = {}
        oldkeys = []
        for k in meta:
            key = k
            if isinstance(k, str):
                for v in context:
                    if '{%s}'%v in k:
                        key = k.replace('{%s}'%v, context[v])
                        oldkeys.append(k)

            newkeys[key] = change_placeholders(meta[k], context)

        for k in oldkeys:
            del meta[k]

        meta.update(newkeys)

    if isinstance(meta, list):
        for i, _ in enumerate(meta):
            meta[i] = change_placeholders(meta[i], context)

    elif isinstance(meta, str):
        for v in context:
            meta = meta.replace('{%s}'%v, context[v])

    return meta

