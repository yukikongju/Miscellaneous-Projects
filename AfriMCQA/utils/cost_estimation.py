def estimate_cost(
    n_requests,
    text_tokens=125,
    image_tokens=70,
    output_tokens=10,
    input_price_per_million=1.75,
    output_price_per_million=14,
):
    """
    - GPT-5.2 at $1.75 / 1M input tokens and $14 / 1M output tokens

    Example Usage:
    > estimate_cost(800, text_token=200, image_tokens=100, output_token=150)
    """
    input_tokens_total = n_requests * (text_tokens + image_tokens)
    output_tokens_total = n_requests * output_tokens

    input_cost = (input_tokens_total / 1_000_000) * input_price_per_million
    output_cost = (output_tokens_total / 1_000_000) * output_price_per_million

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
