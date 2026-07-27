
def next_birthday(date, birthdays):
    
    if not birthdays:
        return (1, 1), []

    sorted_dates = sorted(birthdays.keys())

    for d in sorted_dates:
        if d > date:
            return d, birthdays[d]


    earliest = sorted_dates[0]
    
    return earliest, birthdays[earliest]
