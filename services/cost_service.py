from config import PRICING


def calculate_cost_breakdown(input_tokens=0, output_tokens=0,
                              cache_read_tokens=0, cache_write_tokens=0):
    return {
        "input_cost": round((input_tokens / 1_000_000) * PRICING["input"], 6),
        "output_cost": round((output_tokens / 1_000_000) * PRICING["output"], 6),
        "cache_read_cost": round((cache_read_tokens / 1_000_000) * PRICING["cache_read"], 6),
        "cache_write_cost": round((cache_write_tokens / 1_000_000) * PRICING["cache_write"], 6),
        "total": round(
            (input_tokens / 1_000_000) * PRICING["input"]
            + (output_tokens / 1_000_000) * PRICING["output"]
            + (cache_read_tokens / 1_000_000) * PRICING["cache_read"]
            + (cache_write_tokens / 1_000_000) * PRICING["cache_write"],
            6,
        ),
    }
