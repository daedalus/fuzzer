"""Fast JSON module using orjson when available."""

__all__ = ["JSONDecodeError", "loads", "load", "dumps", "dump"]

try:
    import orjson

    JSONDecodeError = ValueError

    def loads(data):
        return orjson.loads(data)

    def load(fp):
        return orjson.loads(fp.read())

    def dumps(obj, separators=None):
        return orjson.dumps(obj).decode()

    def dump(obj, fp, separators=None):
        fp.write(dumps(obj, separators=separators))

except ImportError:
    from json import JSONDecodeError, dump, dumps, load, loads
