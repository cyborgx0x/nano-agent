value_mapping = {"cotton": 1, "flax": 3, "hemp": 4}


def sort_resource_by_value(array):
    """
    sort the array
    """
    for item in array:
        item["value"] = value_mapping.get(item["name"])
    return sorted(array, key=lambda x: x["value"], reverse=True)

if __name__ == "__main__":
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
    print(sort_resource_by_value(array))
