from datetime import datetime, timedelta


def chunk_date_range(
    start_date: str,
    end_date: str,
    max_chunk_day: int = 30,
    date_format: str = "%Y-%m-%d",
):
    """
    Splits a date range into chunks of up to `chunk_size` days.

    Args:
        start_date (str): Start of the range
        end_date (str): End of the range
        chunk_size (int): Maximum number of days per chunk

    Returns:
        List of tuples: [(chunk_start, chunk_end), ...]
    """
    start_date_datetime = datetime.strptime(start_date, date_format)
    end_date_datetime = datetime.strptime(end_date, date_format)

    if start_date_datetime > end_date_datetime:
        raise ValueError("start_date must be <= end_date")

    chunks = []
    current_start = start_date_datetime

    while current_start <= end_date_datetime:
        current_end = min(current_start + timedelta(days=max_chunk_day - 1), end_date_datetime)
        chunks.append(
            (
                datetime.strftime(current_start, date_format),
                datetime.strftime(current_end, date_format),
            )
        )
        current_start = current_end + timedelta(days=1)

    return chunks
