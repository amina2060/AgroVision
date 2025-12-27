def infection_severity(infection_percent):
    if infection_percent < 20:
        return "Mild"
    elif infection_percent < 50:
        return "Moderate"
    else:
        return "Severe"
