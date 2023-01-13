"""
Based on:
https://stackoverflow.com/questions/47310929/does-python-support-character-type
"""

def progress_bar(progress: float, 
                 total: float, 
                 prefix: str = '',
                 suffix : str = '', 
                 decimals: int = 1, 
                 length: int = 100, 
                 fill: str = '█',
                 blank: str = '-',
                 printEnd: str = "\r" ):
    
    # Determine the progress
    p = progress / total

    # Construct the bar
    fill_length = round(p*length)
    blank_length = length - fill_length
    bar = fill*fill_length + blank*blank_length
    
    # Construct the percentage
    percent_str = ('{0:.' + str(decimals) + 'f}').format(100 * p)

    # Show
    print(f'\r{prefix} |{bar}| {percent_str}% {suffix}', end = printEnd)
    
    # Print New Line on Complete
    if progress >= total: 
        print()