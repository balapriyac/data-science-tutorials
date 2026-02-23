import time
import functools
import hashlib
import json
import logging


# ── 1. @retry ────────────────────────────────────────────────────────────────

def retry(max_attempts=3, delay=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


# ── 2. @timer ─────────────────────────────────────────────────────────────────

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__} completed in {elapsed:.4f}s")
        return result
    return wrapper


# ── 3. @validate_schema ───────────────────────────────────────────────────────

def validate_schema(required_keys):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            missing = required_keys - data.keys()
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
            return func(data, *args, **kwargs)
        return wrapper
    return decorator


# ── 4. @cache_result ──────────────────────────────────────────────────────────

def cache_result(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(
            json.dumps((args, kwargs), sort_keys=True, default=str).encode()
        ).hexdigest()

        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"[CACHE] Computed result for {func.__name__}")
        else:
            print(f"[CACHE] Cache hit for {func.__name__}")

        return cache[key]

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


# ── 5. @log_step ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

def log_step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"START  {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"END    {func.__name__} — OK")
            return result
        except Exception as e:
            logger.error(f"FAIL   {func.__name__} — {type(e).__name__}: {e}")
            raise
    return wrapper


