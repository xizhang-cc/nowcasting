import time

import numpy as np

def naive_persistence(in_precip, observed_precip):
  start = time.time()
  persistence_forecast = np.empty_like(observed_precip)
  for precipitation_index in range(len(observed_precip)):


    # You can use the precipitation observations directly in mm/h for this step.
    if precipitation_index < 1:
      last_observation = in_precip[-1]
    else:
      last_observation = [-1]#observed_precip[precipitation_index-1]

    # last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

    # We set the number of leadtimes (the length of the forecast horizon) to the
    # length of the observed/verification preipitation data. In this way, we'll get
    # a forecast that covers these time intervals.
    n_leadtimes = observed_precip.shape[0]

    # Advect the most recent radar rainfall field and make the nowcast.
    persistence_forecast[precipitation_index] = last_observation

  # This shows the shape of the resulting array with [time intervals, rows, cols]
  print("The shape of the resulting array is: ", persistence_forecast.shape)

  end = time.time()
  print("Advecting the radar rainfall fields took ", (end - start), " seconds")
  return persistence_forecast