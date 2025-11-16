"""
VALUE SORTING MODULE - Resource Prioritization by Economic Value

This module implements an economic decision-making strategy:
prioritize gathering higher-tier (more valuable) resources.

In Albion Online, fiber resources have different tiers:
- Cotton (T2): Low value, common
- Flax (T3): Medium value
- Hemp (T4): High value, rare

The bot can use this to maximize profit per hour by preferring
higher-tier resources when multiple options are available.
"""

# Economic value mapping (higher = more valuable)
# Values are relative weights for prioritization
value_mapping = {"cotton": 1, "flax": 3, "hemp": 4}


def sort_resource_by_value(array):
    """
    Sort detected resources by their economic value (highest first).

    This allows the bot to prioritize gathering valuable resources
    when multiple resources are visible on screen.

    Algorithm:
    1. Add "value" field to each detected resource
    2. Sort by value in descending order (highest first)
    3. Bot will click on array[0] (most valuable resource)

    Args:
        array: List of detected resources from YOLO
               [
                 {"name": "cotton", "confidence": 0.91, "box": {...}},
                 {"name": "hemp", "confidence": 0.85, "box": {...}},
               ]

    Returns:
        list: Same array, sorted by value (hemp before cotton)

    Example:
        Input: [cotton(value=1), hemp(value=4), flax(value=3)]
        Output: [hemp(value=4), flax(value=3), cotton(value=1)]
    """
    # Assign economic value to each resource
    for item in array:
        item["value"] = value_mapping.get(item["name"])

    # Sort by value (highest first)
    return sorted(array, key=lambda x: x["value"], reverse=True)


# ============================================================================
# TEST CODE - Demonstrates value sorting with example detections
# ============================================================================
if __name__ == "__main__":
    # Example: 3 detected resources (2 cotton, 1 flax)
    array = [
        {
            "name": "cotton",
            "class": 1,
            "confidence": 0.9121813178062439,
            "box": {
                "x1": 744.4118041992188,
                "y1": 404.649169921875,
                "x2": 827.045654296875,
                "y2": 477.1366882324219,
            },
        },
        {
            "name": "cotton",
            "class": 1,
            "confidence": 0.9019400477409363,
            "box": {
                "x1": 1176.33154296875,
                "y1": 269.1721496582031,
                "x2": 1239.0404052734375,
                "y2": 337.95953369140625,
            },
        },
        {
            "name": "flax",
            "class": 2,
            "confidence": 0.34675437211990356,
            "box": {
                "x1": 1294.613525390625,
                "y1": 288.8346862792969,
                "x2": 1349.121826171875,
                "y2": 346.90240478515625,
            },
        },
    ]

    # Sort by value - flax (value=3) should come before cotton (value=1)
    sorted_array = sort_resource_by_value(array)
    print("Sorted by value (highest first):")
    for resource in sorted_array:
        print(f"  {resource['name']}: value={resource['value']}, confidence={resource['confidence']:.2f}")
    print("\nBot will click on:", sorted_array[0]["name"])