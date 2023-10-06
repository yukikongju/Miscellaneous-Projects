#  https://en.wikipedia.org/wiki/Haversine_formula

import math

EARTH_RADIUS = 6371000
LATITUDE_SPACING = 0.25
LONGITUDE_SPACING = 0.25

def haversine_distance(lat1, lat2, lon1, lon2):
    """ 
    Function that calculate haversine distance (ie the area covered on earth) 
    in meters

    Parameters
    ----------
    lat1, lat2, lon1, lon2: bounding box of the earth. should be in decimal degrees
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # calculate diagonal distance of bounding box
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    diagonal_distance = EARTH_RADIUS * c

    # calculate area of bounding box
    rectangle_area = diagonal_distance * ((LATITUDE_SPACING + LONGITUDE_SPACING)/2)

    # Calculate the total area covered on Earth in meters
    num_grid_cells = 721 * 1440 * 37
    total_area = rectangle_area * num_grid_cells

    return total_area

    


def main():
    print(haversine_distance(-0.1162, 35.8, -98.2, 42.3))
    

if __name__ == "__main__":
    main()
