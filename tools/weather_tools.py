from dwdweather import DwdWeather
from datetime import date, timedelta
import pandas as pd



def get_dwd_data_from_location(
    lon=7.500833,
    lat=51.242778,
    resolution="daily",
    start=date(2013, 1, 1),
    end=date(2013, 1, 30),
    verbose=0
):
    # iterate through time stamps
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    # Create client object.
    dwd = DwdWeather(resolution=resolution)
    # Find closest station to position.
    closest = dwd.nearest_station(lon=lon, lat=lat)
    out = []
    for single_date in daterange(start, end):
        res = dwd.query(station_id=closest["station_id"], timestamp=single_date)
        if verbose:
            print(res)
        out.append(res)

    out = pd.DataFrame(out)
    out["datetime"] = pd.to_datetime(out["datetime"], format="%Y%m%d")
    return closest, pd.DataFrame(out)