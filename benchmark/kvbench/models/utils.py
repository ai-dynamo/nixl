def get_precision_size(precision: str) -> int:
    if precision == "fp8":
        return 1
    elif precision == "int8":
        return 1
    elif precision == "fp16":
        return 2
    elif precision == "bfloat16":
        return 2
    else:
        raise ValueError(f"Unsupported precision: {precision}")