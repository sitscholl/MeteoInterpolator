import datetime
import pandas as pd

def split_dates(start_date, end_date, freq, n_days=7, split_on_year=False):
    """
    freq: string (e.g., '1h', '15min') or pd.Timedelta
    """
    if end_date < start_date:
        raise ValueError(f"Start date cannot be smaller than end date. Got {start_date} and {end_date}")

    # Convert freq to a Timedelta for easy math
    freq_delta = pd.Timedelta(freq)
    date_pairs = []
    current_start = start_date
    
    while current_start <= end_date:
        # 1. Calculate the normal step (e.g., 7 days)
        # Note: We subtract one frequency unit from the potential end 
        # so that the chunk spans n_days TOTAL including the start and end.
        potential_end = current_start + datetime.timedelta(days=n_days) - freq_delta
        
        if split_on_year:
            # 2. Calculate the very last possible timestamp of the current year
            # (December 31st, 23:59:59... or whatever the last 'freq' step is)
            next_year_start = datetime.datetime(current_start.year + 1, 1, 1, tzinfo=current_start.tzinfo)
            last_of_year = next_year_start - freq_delta
            
            # 3. Pick the earliest of the three boundaries
            current_end = min(potential_end, end_date, last_of_year)
        else:
            current_end = min(potential_end, end_date)

        # Safety: check we didn't go backwards
        if current_end < current_start:
            # This can happen if start_date is already the last timestamp of the year
            # Force the end to be the start so we at least get one record
            current_end = current_start

        date_pairs.append((current_start, current_end))
        
        # 4. MOVE TO THE NEXT TIMESTAMP
        # The next start is exactly one frequency step after the current end
        current_start = current_end + freq_delta
    
    return date_pairs